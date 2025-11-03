# Job Search Chatbot

Interactive Streamlit assistant that orchestrates Model Context Protocol (MCP) servers to help job seekers discover openings, tailor application collateral, and track career goals.

## Key Features
- Conversational UI built with Streamlit for discovering roles, companies, and salary trends
- MCP servers for resume/cover letter generation (`docx-creator`) and job aggregation (`jobspy-server`)
- Memory store for short-term conversation recall and long-term context using LangMem
- Supports Azure OpenAI or OpenAI completions with a single configuration switch
- Automatically formats downloadable `.docx` documents ready for submission

## Architecture
- **Client (`Clients/mcp_streamlit.py`)**: Streamlit front-end that hosts the chat interface, initializes the REACT agent, and manages session state.
- **Servers (`Servers/`)**:
  - `docx_server.py`: Generates resumes, cover letters, and formatted documents via `python-docx`.
  - `jobspy_server.py`: Scrapes job listings from Indeed and LinkedIn using `jobspy` and summarizes results.
  - `server_capabilities.json` & `servers_config.json`: Descriptions and connection metadata consumed by the MCP client.
- **Agent runtime**: `langgraph` REACT agent with MCP tools, LangMem memory store, and embeddings derived from Azure or OpenAI APIs.

## Prerequisites
- Python 3.10+
- Node.js 18+ (required for the filesystem MCP server invoked via `npx`)
- An OpenAI API key *or* Azure OpenAI deployment credentials

## Environment Variables
Create a `.env` file in the repository root and populate the keys that match your setup. When both Azure and OpenAI values are present, Azure is preferred.

```
# OpenAI (fallback)
OPENAI_API_KEY=sk-...

# Azure OpenAI (preferred)
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-3-small
```

Optional: set `FILESYSTEM_ROOT` to point the filesystem MCP server at a specific directory (defaults to the repository root).

## Local Development
1. Create and activate a virtual environment.
	```bash
	python3 -m venv .venv
	source .venv/bin/activate
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

## Docker Usage
1. Build the image.
	```bash
	docker build -t job-search-chatbot .
	```
2. Run the container, mounting your `.env` file for secrets.
	```bash
	docker run --env-file .env -p 8501:8501 job-search-chatbot
	```

The container installs Node.js and launches the Streamlit server. The MCP subprocesses start automatically when the agent initializes.

## Project Structure
```
Clients/
  mcp_streamlit.py      # Streamlit UI and agent runtime
Servers/
  docx_server.py        # FastMCP DOCX generation server
  jobspy_server.py      # FastMCP job scraping server
  server_capabilities.json
  servers_config.json
requirements.txt
README.md
```

## Troubleshooting
- **Missing tool errors**: Confirm MCP servers defined in `Servers/servers_config.json` can be executed inside your environment (Python entrypoints + Node.js filesystem server).
- **Authentication failures**: Ensure `.env` values are loaded (restart Streamlit after updates).
- **Document generation issues**: Verify `python-docx` is installed and you have permissions to write to the destination path supplied to MCP tools.
