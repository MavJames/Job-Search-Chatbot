#!/bin/bash

# Azure Container Apps deployment script for Job Search Chatbot
# This script deploys the application to Azure Container Apps

set -e

# Configuration
RESOURCE_GROUP="${RESOURCE_GROUP:-job-search-rg}"
LOCATION="${LOCATION:-eastus}"
CONTAINER_APP_NAME="${CONTAINER_APP_NAME:-job-search-chatbot}"
CONTAINER_REGISTRY="${CONTAINER_REGISTRY:-jobsearchcr}"
IMAGE_NAME="job-search-chatbot"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "🚀 Deploying Job Search Chatbot to Azure Container Apps"
echo "Resource Group: $RESOURCE_GROUP"
echo "Location: $LOCATION"
echo "Container App: $CONTAINER_APP_NAME"

# Check if logged in to Azure
echo "Checking Azure login status..."
az account show > /dev/null 2>&1 || {
    echo "❌ Not logged in to Azure. Please run 'az login' first."
    exit 1
}

# Create resource group if it doesn't exist
echo "Creating resource group..."
az group create \
    --name "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output none

# Create Azure Container Registry if it doesn't exist
echo "Creating Azure Container Registry..."
az acr create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$CONTAINER_REGISTRY" \
    --sku Basic \
    --admin-enabled true \
    --output none || echo "Registry already exists"

# Get ACR credentials
ACR_LOGIN_SERVER=$(az acr show --name "$CONTAINER_REGISTRY" --query loginServer --output tsv)
ACR_USERNAME=$(az acr credential show --name "$CONTAINER_REGISTRY" --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name "$CONTAINER_REGISTRY" --query passwords[0].value --output tsv)

echo "ACR Login Server: $ACR_LOGIN_SERVER"

# Build and push Docker image
echo "Building Docker image..."
docker build -t "$IMAGE_NAME:$IMAGE_TAG" .

echo "Tagging image for ACR..."
docker tag "$IMAGE_NAME:$IMAGE_TAG" "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG"

echo "Logging in to ACR..."
echo "$ACR_PASSWORD" | docker login "$ACR_LOGIN_SERVER" --username "$ACR_USERNAME" --password-stdin

echo "Pushing image to ACR..."
docker push "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG"

# Create Container Apps environment
ENVIRONMENT_NAME="${CONTAINER_APP_NAME}-env"
echo "Creating Container Apps environment..."
az containerapp env create \
    --name "$ENVIRONMENT_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output none || echo "Environment already exists"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please create one with your configuration."
    exit 1
fi

# Load environment variables from .env
source .env

# Create or update the container app
echo "Deploying container app..."
az containerapp create \
    --name "$CONTAINER_APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --environment "$ENVIRONMENT_NAME" \
    --image "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG" \
    --registry-server "$ACR_LOGIN_SERVER" \
    --registry-username "$ACR_USERNAME" \
    --registry-password "$ACR_PASSWORD" \
    --target-port 3000 \
    --ingress external \
    --min-replicas 1 \
    --max-replicas 3 \
    --cpu 1.0 \
    --memory 2.0Gi \
    --secrets \
        openai-api-key="$OPENAI_API_KEY" \
        azure-openai-api-key="$AZURE_OPENAI_API_KEY" \
        azure-openai-endpoint="$AZURE_OPENAI_ENDPOINT" \
        slack-bot-token="$SLACK_BOT_TOKEN" \
        slack-signing-secret="$SLACK_SIGNING_SECRET" \
        slack-app-token="$SLACK_APP_TOKEN" \
    --env-vars \
        AZURE_OPENAI_API_VERSION="$AZURE_OPENAI_API_VERSION" \
        AZURE_OPENAI_DEPLOYMENT="$AZURE_OPENAI_DEPLOYMENT" \
        AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT="$AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT" \
        OPENAI_API_KEY=secretref:openai-api-key \
        AZURE_OPENAI_API_KEY=secretref:azure-openai-api-key \
        AZURE_OPENAI_ENDPOINT=secretref:azure-openai-endpoint \
        SLACK_BOT_TOKEN=secretref:slack-bot-token \
        SLACK_SIGNING_SECRET=secretref:slack-signing-secret \
        SLACK_APP_TOKEN=secretref:slack-app-token \
    --output none || {
        echo "Updating existing container app..."
        az containerapp update \
            --name "$CONTAINER_APP_NAME" \
            --resource-group "$RESOURCE_GROUP" \
            --image "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG" \
            --output none
    }

# Get the app URL
APP_URL=$(az containerapp show \
    --name "$CONTAINER_APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query properties.configuration.ingress.fqdn \
    --output tsv)

echo ""
echo "✅ Deployment complete!"
echo "🌐 Application URL: https://$APP_URL"
echo ""
echo "To view logs, run:"
echo "az containerapp logs show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --follow"
