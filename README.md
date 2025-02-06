# MCP servers

CatalystNeuro maintained Model Context Protocol (MCP) servers.

## Available servers

- Semantic search
- ...


## Usage with cline

To install a server:

```bash
cd servers/<server_name>
python -m venv .venv && .venv/bin/pip install -e .
```

Then configure the server in your `cline_mcp_settings.json`:

```json
{
  "mcpServers": {
    "<server_name": {
      "command": "/path_to/servers/<server_name>/.venv/bin/python",
      "args": [
        "/path_to/servers/<server_name>/src/main.py"
      ],
      "env": {
        "ENV_VAR_0": "VALUE0",
        "ENV_VAR_1": "VALUE1",
      },
      "disabled": false,
      "autoApprove": []
    }
}
```

Each server has its own configuration, so you need to check the server's documentation for the required environment variables.


## Development

To quickly run and test a server, you can use:

```bash
# export the required environment variables
export OPENAI_API_KEY="your-openai-api-key"

cd servers/<server_name>
mcp dev src/main.py

# or for development
mcp dev src/main.py --with-editable .
```

