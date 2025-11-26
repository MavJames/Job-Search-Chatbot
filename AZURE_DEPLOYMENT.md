# Azure Deployment Guide

This guide explains how to deploy the Job Search Chatbot to Azure Container Apps.

## Prerequisites

1. **Azure Account**: Active Azure subscription
2. **Azure CLI**: Install from https://docs.microsoft.com/cli/azure/install-azure-cli
3. **Docker**: Install from https://docs.docker.com/get-docker/
4. **Environment Variables**: Complete `.env` file with all required credentials

## Environment Variables Required

Create a `.env` file in the project root with the following variables:

```bash
# OpenAI Configuration (fallback)
OPENAI_API_KEY=sk-...

# Azure OpenAI Configuration (preferred)
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-3-small

# Slack Configuration
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...
SLACK_APP_TOKEN=xapp-...  # For Socket Mode
```

## Deployment Methods

### Method 1: Automated Deployment Script (Recommended)

The easiest way to deploy is using the provided deployment script:

```bash
# Login to Azure
az login

# Run the deployment script
./deploy-azure.sh
```

The script will:
1. Create a resource group
2. Set up Azure Container Registry
3. Build and push your Docker image
4. Create Container Apps environment
5. Deploy the application
6. Output the application URL

#### Customizing the Deployment

You can customize the deployment by setting environment variables:

```bash
# Custom resource group and location
export RESOURCE_GROUP="my-job-search-rg"
export LOCATION="westus2"
export CONTAINER_APP_NAME="my-job-bot"
export CONTAINER_REGISTRY="myjobsearchcr"

./deploy-azure.sh
```

### Method 2: Manual Deployment

If you prefer manual control:

#### Step 1: Create Resource Group

```bash
az group create \
    --name job-search-rg \
    --location eastus
```

#### Step 2: Create Container Registry

```bash
az acr create \
    --resource-group job-search-rg \
    --name jobsearchcr \
    --sku Basic \
    --admin-enabled true
```

#### Step 3: Build and Push Image

```bash
# Build the image
docker build -t job-search-chatbot:latest .

# Get ACR login server
ACR_LOGIN_SERVER=$(az acr show --name jobsearchcr --query loginServer --output tsv)

# Tag the image
docker tag job-search-chatbot:latest $ACR_LOGIN_SERVER/job-search-chatbot:latest

# Login to ACR
az acr login --name jobsearchcr

# Push the image
docker push $ACR_LOGIN_SERVER/job-search-chatbot:latest
```

#### Step 4: Create Container Apps Environment

```bash
az containerapp env create \
    --name job-search-env \
    --resource-group job-search-rg \
    --location eastus
```

#### Step 5: Deploy Container App

```bash
az containerapp create \
    --name job-search-chatbot \
    --resource-group job-search-rg \
    --environment job-search-env \
    --image $ACR_LOGIN_SERVER/job-search-chatbot:latest \
    --registry-server $ACR_LOGIN_SERVER \
    --target-port 3000 \
    --ingress external \
    --min-replicas 1 \
    --max-replicas 3 \
    --cpu 1.0 \
    --memory 2.0Gi \
    --secrets \
        openai-api-key="YOUR_OPENAI_KEY" \
        azure-openai-api-key="YOUR_AZURE_KEY" \
        slack-bot-token="YOUR_SLACK_TOKEN" \
    --env-vars \
        OPENAI_API_KEY=secretref:openai-api-key \
        AZURE_OPENAI_API_KEY=secretref:azure-openai-api-key \
        SLACK_BOT_TOKEN=secretref:slack-bot-token
```

### Method 3: Using Docker Compose (Local Testing)

For local testing before deploying to Azure:

```bash
# Start the services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the services
docker-compose down
```

## Configuration Options

### Slack Integration

The bot supports two modes for Slack:

1. **Socket Mode** (Easier, recommended for testing):
   - Set `SLACK_APP_TOKEN` in environment variables
   - No public URL required
   - Bot connects to Slack via WebSocket

2. **HTTP Mode** (For production):
   - Don't set `SLACK_APP_TOKEN`
   - Requires public URL (provided by Azure Container Apps)
   - Configure Slack Event Subscriptions URL: `https://your-app.azurecontainerapps.io/slack/events`

### MCP Server Configuration

The application uses `servers_config_slack.json` which includes:
- `docx_server`: Creates resumes and cover letters (returns base64 content)
- `jobspy-server`: Searches for jobs on Indeed and LinkedIn

The filesystem server is excluded from the Slack configuration to avoid file I/O dependencies.

## Monitoring and Troubleshooting

### View Logs

```bash
# Real-time logs
az containerapp logs show \
    --name job-search-chatbot \
    --resource-group job-search-rg \
    --follow

# Recent logs
az containerapp logs show \
    --name job-search-chatbot \
    --resource-group job-search-rg \
    --tail 100
```

### Check Application Status

```bash
az containerapp show \
    --name job-search-chatbot \
    --resource-group job-search-rg \
    --query properties.runningStatus
```

### Update Application

To deploy updates:

```bash
# Rebuild and push image
docker build -t job-search-chatbot:latest .
docker tag job-search-chatbot:latest $ACR_LOGIN_SERVER/job-search-chatbot:latest
docker push $ACR_LOGIN_SERVER/job-search-chatbot:latest

# Update the container app
az containerapp update \
    --name job-search-chatbot \
    --resource-group job-search-rg \
    --image $ACR_LOGIN_SERVER/job-search-chatbot:latest
```

Or simply run the deployment script again:
```bash
./deploy-azure.sh
```

### Scale Application

```bash
# Scale up
az containerapp update \
    --name job-search-chatbot \
    --resource-group job-search-rg \
    --min-replicas 2 \
    --max-replicas 5

# Scale down (for cost savings)
az containerapp update \
    --name job-search-chatbot \
    --resource-group job-search-rg \
    --min-replicas 0 \
    --max-replicas 1
```

## Cost Optimization

To minimize Azure costs:

1. **Use consumption-only plan**: Set `--min-replicas 0`
2. **Choose appropriate region**: Some regions are cheaper
3. **Right-size resources**: Start with lower CPU/memory and scale up if needed
4. **Set up auto-scaling**: Only scale when needed

```bash
az containerapp update \
    --name job-search-chatbot \
    --resource-group job-search-rg \
    --min-replicas 0 \
    --max-replicas 2 \
    --cpu 0.5 \
    --memory 1.0Gi
```

## Security Best Practices

1. **Use Secrets**: Never hardcode credentials in code
2. **Enable HTTPS**: Always use HTTPS (default in Container Apps)
3. **Restrict Access**: Use Azure networking features to limit access
4. **Rotate Keys**: Regularly rotate API keys and tokens
5. **Monitor Logs**: Enable Azure Monitor for security alerts

## Cleanup

To remove all Azure resources:

```bash
az group delete \
    --name job-search-rg \
    --yes \
    --no-wait
```

## Support

For issues:
1. Check logs using the commands above
2. Verify environment variables are set correctly
3. Ensure all Slack tokens and API keys are valid
4. Check Azure Container Apps documentation: https://learn.microsoft.com/azure/container-apps/

## Additional Resources

- [Azure Container Apps Documentation](https://learn.microsoft.com/azure/container-apps/)
- [Slack API Documentation](https://api.slack.com/)
- [Azure OpenAI Documentation](https://learn.microsoft.com/azure/ai-services/openai/)
