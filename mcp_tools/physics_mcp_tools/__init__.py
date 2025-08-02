"""Physics Forces MCP Tool - Convenience methods to start servers."""

import click


@click.command()
@click.option('--run', 'command', default='forces-server', help='Command to run')
@click.option(
    '--host',
    'host',
    default='localhost',
    help='Host on which the server is started or the client connects to',
)
@click.option(
    '--port',
    'port',
    default=10100,
    help='Port on which the server is started or the client connects to',
)
@click.option(
    '--transport',
    'transport',
    default='streamable_http',
    help='MCP Transport (stdio, sse, or streamable_http)',
)
def main(command, host, port, transport) -> None:
    """Main entry point for the Physics Forces MCP server."""
    if command == 'forces-server':
        from physics_mcp_tools.forces_mcp_server import serve
        serve(host, port, transport)
    elif command == 'kinematics-server':
        from physics_mcp_tools.kinematics_mcp_server import serve
        serve(host, port, transport)
    elif command == 'math-server':
        from physics_mcp_tools.math_mcp_server import serve
        serve(host, port, transport)
    elif command == 'momentum-server':
        from physics_mcp_tools.momentum_mcp_server import serve
        serve(host, port, transport)
    elif command == 'energy-server':
        from physics_mcp_tools.energy_mcp_server import serve
        serve(host, port, transport)
    elif command == 'circuit-server':
        from physics_mcp_tools.circuit_mcp_server import serve
        serve(host, port, transport)
    else:
        raise ValueError(f'Unknown run option: {command}')


if __name__ == "__main__":
    main()
