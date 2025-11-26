# Quick Reference Guide

## 🚀 Quick Start

### Streamlit (Web Interface)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run Clients/mcp_streamlit.py
```

### Slack Bot (Local)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python Clients/mcp_slack.py
```

### Docker (Local)
```bash
docker-compose up -d
docker-compose logs -f
```

### Azure (Production)
```bash
az login
./deploy-azure.sh
```

## 📝 Configuration Files

| File | Purpose | Used By |
|------|---------|---------|
| `servers_config.json` | MCP servers with filesystem | Streamlit |
| `servers_config_slack.json` | MCP servers without filesystem | Slack Bot |
| `server_capabilities.json` | Server descriptions | Both |
| `.env` | Environment variables | All |
| `.env.example` | Environment template | Documentation |

## 🔧 Environment Variables

### Required (Choose One)
```bash
# Option 1: OpenAI
OPENAI_API_KEY=sk-...

# Option 2: Azure OpenAI (Preferred)
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-3-small
```

### Slack Bot (Required for Slack)
```bash
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...
SLACK_APP_TOKEN=xapp-...  # For Socket Mode
```

## 🐳 Docker Commands

```bash
# Build image
docker build -t job-search-chatbot .

# Run with env file
docker run --env-file .env -p 3000:3000 job-search-chatbot

# Docker Compose
docker-compose up -d        # Start
docker-compose down         # Stop
docker-compose logs -f      # View logs
docker-compose restart      # Restart
```

## ☁️ Azure Commands

```bash
# Deploy (automated)
./deploy-azure.sh

# View logs
az containerapp logs show \
  --name job-search-chatbot \
  --resource-group job-search-rg \
  --follow

# Update app
az containerapp update \
  --name job-search-chatbot \
  --resource-group job-search-rg \
  --image <new-image>

# Scale
az containerapp update \
  --name job-search-chatbot \
  --resource-group job-search-rg \
  --min-replicas 0 \
  --max-replicas 2

# Delete resources
az group delete \
  --name job-search-rg \
  --yes
```

## 🔍 Troubleshooting

### Check Python Environment
```bash
python --version  # Should be 3.11+
pip list | grep -E "fastmcp|langchain|slack"
```

### Test MCP Servers
```bash
# Test docx server
python Servers/docx_server.py

# Test jobspy server
python Servers/jobspy_server.py
```

### Check Configuration
```bash
# Verify .env file
cat .env

# Test config files are valid JSON
python -m json.tool Servers/servers_config.json
python -m json.tool Servers/servers_config_slack.json
```

### View Logs
```bash
# Local Python logs
python Clients/mcp_slack.py 2>&1 | tee app.log

# Docker logs
docker logs <container-id> --follow

# Azure logs
az containerapp logs show --name job-search-chatbot --resource-group job-search-rg --follow
```

## 📊 Document Flow

### Streamlit (with filesystem)
```
User Request → Agent → docx_server → Save to file → Return path → Display download button
```

### Slack Bot (no filesystem)
```
User Request → Agent → docx_server → Return base64 → Decode → Upload to Slack
```

## 🛠️ Common Tasks

### Add New MCP Server
1. Create server in `Servers/your_server.py`
2. Add to `servers_config.json` or `servers_config_slack.json`
3. Add description to `server_capabilities.json`
4. Restart application

### Update Dependencies
```bash
pip install --upgrade -r requirements.txt
pip freeze > requirements.txt
```

### Change Python Version
1. Update `.python-version` file
2. Update `Dockerfile` base image
3. Update `.devcontainer/devcontainer.json`
4. Rebuild containers

### Modify Document Format
Edit `Servers/docx_server.py`:
- `add_section_heading()` - Heading styles
- `create_resume()` - Resume layout
- `create_cover_letter()` - Cover letter layout

## 📚 File Locations

```
Project Root/
├── Clients/
│   ├── mcp_streamlit.py          # Web interface
│   └── mcp_slack.py               # Slack bot
├── Servers/
│   ├── docx_server.py             # Document generation
│   ├── jobspy_server.py           # Job search
│   ├── servers_config.json        # Streamlit config
│   ├── servers_config_slack.json  # Slack config
│   └── server_capabilities.json   # Descriptions
├── .env                           # Your secrets (create this)
├── .env.example                   # Template
├── requirements.txt               # Dependencies
├── Dockerfile                     # Container def
├── docker-compose.yml             # Docker Compose
├── deploy-azure.sh               # Azure script
├── README.md                      # Main docs
├── AZURE_DEPLOYMENT.md           # Azure guide
└── IMPLEMENTATION_SUMMARY.md     # Changes log
```

## 🔗 Useful Links

- [Slack API](https://api.slack.com/)
- [Azure Container Apps](https://learn.microsoft.com/azure/container-apps/)
- [Azure OpenAI](https://learn.microsoft.com/azure/ai-services/openai/)
- [FastMCP](https://github.com/jlowin/fastmcp)
- [LangGraph](https://langchain-ai.github.io/langgraph/)

## 💡 Tips

1. **Use Socket Mode for Slack** during development (easier setup)
2. **Test locally** with Docker Compose before Azure deployment
3. **Monitor costs** in Azure - use `--min-replicas 0` when not in use
4. **Keep secrets safe** - never commit `.env` file
5. **Update regularly** - check for dependency updates monthly

## 🆘 Getting Help

1. Check `README.md` for general usage
2. See `AZURE_DEPLOYMENT.md` for Azure issues
3. Review `IMPLEMENTATION_SUMMARY.md` for recent changes
4. Check application logs for error messages
5. Verify environment variables are set correctly
