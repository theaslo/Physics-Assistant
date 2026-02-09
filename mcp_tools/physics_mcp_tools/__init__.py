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
    elif command == 'angular-motion-server':
        from physics_mcp_tools.angular_motion_mcp_server import serve
        serve(host, port, transport)
    elif command == 'circuit-server':
        from physics_mcp_tools.circuit_mcp_server import serve
        serve(host, port, transport)
    elif command == 'thermodynamics-server':
        from physics_mcp_tools.thermodynamics_mcp_server import serve
        serve(host, port, transport)
    elif command == 'waves-server':
        from physics_mcp_tools.waves_mcp_server import serve
        serve(host, port, transport)
    elif command == 'electromagnetism-server':
        from physics_mcp_tools.electromagnetism_mcp_server import serve
        serve(host, port, transport)
    elif command == 'optics-server':
        from physics_mcp_tools.optics_mcp_server import serve
        serve(host, port, transport)
    elif command == 'modern-physics-server':
        from physics_mcp_tools.modern_physics_mcp_server import serve
        serve(host, port, transport)
    else:
        raise ValueError(f'Unknown run option: {command}')


if __name__ == "__main__":
    main()
