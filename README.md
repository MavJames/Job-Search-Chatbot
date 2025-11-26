# Job Search Chatbot

Interactive assistant that orchestrates Model Context Protocol (MCP) servers to help job seekers discover openings, tailor application collateral, and track career goals. Available as both Streamlit web app and Slack bot.

## Key Features
- Conversational UI built with Streamlit or Slack for discovering roles, companies, and salary trends
- MCP servers for resume/cover letter generation (`docx-creator`) and job aggregation (`jobspy-server`)
- Documents returned as base64 for seamless Slack integration (no filesystem dependencies)
- Memory store for short-term conversation recall and long-term context using LangMem
- Supports Azure OpenAI or OpenAI completions with a single configuration switch
- Automatically formats downloadable `.docx` documents ready for submission
- Deployable to Azure Container Apps for production use

## Architecture
- **Clients**:
  - `Clients/mcp_streamlit.py`: Streamlit front-end for web interface
  - `Clients/mcp_slack.py`: Slack bot integration (uses Socket Mode or HTTP Mode)
- **Servers (`Servers/`)**:
  - `docx_server.py`: Generates resumes, cover letters, and formatted documents via `python-docx`. Returns documents as base64 encoded strings.
  - `jobspy_server.py`: Scrapes job listings from Indeed and LinkedIn using `jobspy` and summarizes results.
  - `servers_config.json`: Configuration for Streamlit client (includes filesystem server)
  - `servers_config_slack.json`: Configuration for Slack client (no filesystem server)
  - `server_capabilities.json`: Descriptions of server capabilities consumed by clients
- **Agent runtime**: `langgraph` REACT agent with MCP tools, LangMem memory store, and embeddings derived from Azure or OpenAI APIs.

## Prerequisites
- Python 3.10+
- Node.js 18+ (required for the filesystem MCP server invoked via `npx`)
- An OpenAI API key *or* Azure OpenAI deployment credentials

## Environment Variables
Create a `.env` file in the repository root and populate the keys that match your setup. When both Azure and OpenAI values are present, Azure is preferred.

```bash
# OpenAI (fallback)
OPENAI_API_KEY=sk-...

# Azure OpenAI (preferred)
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-3-small

# Slack (only needed for Slack bot)
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...
SLACK_APP_TOKEN=xapp-...  # For Socket Mode (recommended for development)
```

Optional: set `FILESYSTEM_ROOT` to point the filesystem MCP server at a specific directory (defaults to the repository root). Note: The Slack client configuration excludes the filesystem server.

## Local Development

### Streamlit Web App

1. Create and activate a virtual environment.
	```bash
	python3 -m venv .venv
	source .venv/bin/activate  # On macOS/Linux
	# or
	.venv\Scripts\activate  # On Windows
	```
2. Install dependencies.
	```bash
	pip install -r requirements.txt
	```
3. Ensure Node.js is available (for the filesystem MCP server invoked through `npx`).
4. Launch the Streamlit client. MCP servers are started on demand by the client.
	```bash
	streamlit run Clients/mcp_streamlit.py
	```

The app will open at `http://localhost:8501`. Upload a resume (PDF or DOCX) to seed the memory system and start the conversation.

### Slack Bot

1. Set up a Slack app at https://api.slack.com/apps with the following:
   - Bot Token Scopes: `chat:write`, `files:write`, `files:read`, `im:history`, `im:read`, `im:write`
   - Event Subscriptions: `message.im`, `file_shared`
   - Enable Socket Mode and create an App-Level Token

2. Create `.env` file with Slack credentials (see Environment Variables section above)

3. Run the Slack bot:
	```bash
	python Clients/mcp_slack.py
	```

The bot will connect to Slack via Socket Mode. Send it a direct message to start chatting!

## Docker Usage

### Local Docker Development
1. Build the image.
	```bash
	docker build -t job-search-chatbot .
	```
2. Run the container, mounting your `.env` file for secrets.
	```bash
	docker run --env-file .env -p 8501:8501 job-search-chatbot
	```

Alternatively, use Docker Compose:
```bash
docker-compose up -d
```

The container installs Node.js and launches the application. The MCP subprocesses start automatically when the agent initializes.

## Azure Deployment

For production deployment to Azure Container Apps, see the comprehensive guide:

📖 **[Azure Deployment Guide](AZURE_DEPLOYMENT.md)**

Quick deployment:
```bash
az login
./deploy-azure.sh
```

This will:
- Create Azure Container Registry
- Build and push Docker image
- Deploy to Azure Container Apps
- Output your application URL

## Project Structure
```
Clients/
  mcp_streamlit.py          # Streamlit web UI and agent runtime
  mcp_slack.py              # Slack bot integration
Servers/
  docx_server.py            # FastMCP DOCX generation server (returns base64)
  jobspy_server.py          # FastMCP job scraping server
  servers_config.json       # Config for Streamlit (with filesystem)
  servers_config_slack.json # Config for Slack (without filesystem)
  server_capabilities.json  # Server capability descriptions
requirements.txt
README.md
AZURE_DEPLOYMENT.md         # Azure deployment guide
Dockerfile                  # Container definition
docker-compose.yml          # Docker Compose configuration
deploy-azure.sh             # Azure deployment script
azure-deployment.yaml       # Kubernetes/Container Apps manifest
.python-version             # Python version specification
```

## Troubleshooting
- **Missing tool errors**: Confirm MCP servers defined in config files can be executed inside your environment (Python entrypoints + Node.js filesystem server for Streamlit).
- **Authentication failures**: Ensure `.env` values are loaded (restart application after updates).
- **Document generation issues**: 
  - For Streamlit: Verify `python-docx` is installed and you have permissions to write to the destination path.
  - For Slack: Documents are returned as base64 and uploaded directly to Slack - no filesystem access needed.
- **Slack connection issues**: Ensure `SLACK_APP_TOKEN` is set for Socket Mode, or configure Event Subscriptions URL for HTTP Mode.
- **Azure deployment issues**: Check the [Azure Deployment Guide](AZURE_DEPLOYMENT.md) for detailed troubleshooting steps.

## Configuration Notes

### Streamlit vs Slack Configurations
- **Streamlit** (`servers_config.json`): Includes filesystem server for local file operations
- **Slack** (`servers_config_slack.json`): Excludes filesystem server; documents handled via base64

### Document Handling
- Documents are created by `docx_server.py` and returned as base64 encoded strings
- Slack client automatically decodes and uploads to Slack channels
- No temporary file storage required for Slack integration
