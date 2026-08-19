# MCP Tools Status

The `physics-mcp` CLI now routes all supported subject servers through
`physics_mcp_tools.__init__.py`.

Supported servers:
- `forces-server`
- `kinematics-server`
- `circuit-server`
- `math-server`
- `momentum-server`
- `energy-server`
- `angular-motion-server`
- `thermodynamics-server`
- `waves-server`
- `electromagnetism-server`
- `optics-server`
- `modern-physics-server`

Current verification command:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

The circuit placeholder has been replaced with real circuit tools, and the
waves, electromagnetism, optics, and circuit servers expose MCP factories for
direct tool-level tests.
