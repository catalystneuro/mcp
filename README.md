# MCP servers

CatalystNeuro maintained Model Context Protocol (MCP) servers.

## Available servers

- Semantic search
- ...


## Usage with cline

add instructions here


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

