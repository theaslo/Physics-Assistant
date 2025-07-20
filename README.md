# Physics-Assistant

A comprehensive physics education tool featuring MCP (Model Context Protocol) integration for force calculations and analysis.

## Features

### MCP Tools for Physics Forces
- **1D Force Addition**: Calculate net forces along a single axis
- **2D Force Addition**: Vector addition with magnitude and direction
- **Force Component Resolution**: Break forces into x and y components
- **Equilibrium Analysis**: Check force balance and suggest corrections
- **Free Body Diagrams**: Generate text-based force diagrams
- **Spring Forces**: Hooke's Law calculations
- **Friction Forces**: Static and kinetic friction analysis
- **Inclined Plane Analysis**: Forces on angled surfaces
- **Tension Forces**: Rope and pulley system analysis
- **Vector Operations**: Addition, subtraction, dot product, cross product

## Quick Start

### Installation
```bash
cd mcp_tools
uv python pin 3.13
uv sync
```

### Running the MCP Server
```bash
# Start with default settings (localhost:10100, streamable_http)
python -m mcp_tools

# Custom configuration
python -m mcp_tools --host 0.0.0.0 --port 8000 --transport stdio
```

### Available Transports
- `streamable_http` (default): HTTP-based streaming transport
- `sse`: Server-Sent Events transport  
- `stdio`: Standard input/output transport

## Usage Examples

The MCP server provides physics calculation tools that can be integrated with AI assistants and other applications supporting the Model Context Protocol.

### Example Force Calculations
- Add forces: `[{"magnitude": 10, "angle": 30}, {"magnitude": 15, "angle": 120}]`
- Analyze equilibrium and get balancing force recommendations
- Calculate spring forces using Hooke's Law
- Determine friction forces for different surface types

## Development

### Dependencies
- Python 3.11+
- FastMCP framework
- Click for CLI interface
- Uvicorn for HTTP transport

### Project Structure
```
mcp_tools/
├── __init__.py          # Main entry point and CLI
├── forces_mcp.py        # FastMCP force calculation tools
├── server.py            # Server implementation
└── pyproject.toml       # Project configuration
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

Asli Tandogan-Kunkel (doogers@uconn.edu)
