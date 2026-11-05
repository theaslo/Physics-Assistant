"""Conceptual MCP tools: ground dialogue without inventing numerical inputs."""

from physics_mcp_tools.database_logger import DatabaseLogger, create_tool_wrapper


PRINCIPLES = {
    "momentum": (
        "Linear momentum is the vector p = m v. Choose a positive direction and keep velocity signs. "
        "Net external impulse J = integral(F_ext dt) = Delta p; for constant average force J = F_avg Delta t. "
        "Momentum of a defined system is conserved only when its net external impulse is negligible. "
        "Internal forces can change each object's momentum without changing total system momentum. "
        "In a perfectly inelastic collision the objects stick together: "
        "m1 v1i + m2 v2i = (m1 + m2) vf for an isolated 1D system. "
        "The two-object system has the SAME total mass m1 + m2 before and after sticking; "
        "both objects belong to the system even when one is initially at rest. Momentum is redistributed "
        "between the objects, not reduced by an increase in total system mass. "
        "Their common final velocity is the mass-weighted average of their initial velocities. "
        "For an initially stationary second object, vf = [m1/(m1 + m2)] v1i: the original momentum "
        "is shared by both objects moving together, so the common speed is below the first object's initial speed. "
        "Do not generalize this speed decrease to arbitrary initial velocities. "
        "Elastic collisions conserve both total momentum and kinetic energy. "
        "Inelastic collisions need not conserve kinetic energy, but total energy is still conserved. "
        "Kinetic energy is scalar K = (1/2) m v^2, not momentum. "
        "Momentum units are kg m/s; impulse units are N s. Equal and opposite momenta can sum to zero "
        "while the objects each have nonzero kinetic energy. Use numerical MCP tools to calculate numbers."
    ),
    "energy": (
        "Work is W = integral(F dot dr); for a constant force W = F d cos(theta), "
        "where theta is the angle between force and displacement. Opposite directions give negative work; "
        "perpendicular forces do zero work. Net work equals change in kinetic energy: W_net = Kf - Ki. "
        "K = (1/2) m v^2 is a nonnegative scalar and depends on speed squared. "
        "Near Earth Ug = m g h relative to a chosen reference; changes are independent of that reference. "
        "An ideal spring stores Us = (1/2) k x^2, with x measured from equilibrium. "
        "Mechanical energy K + U is conserved when the relevant nonconservative work is zero. "
        "More generally Delta(K + U) = W_nonconservative for the chosen system. "
        "Friction can convert mechanical energy to internal thermal energy; total energy is not destroyed. "
        "Average power P = W/Delta t, instantaneous power P = F dot v. Energy units are J, power W. "
        "State the system, reference level, endpoints, and assumptions before applying conservation. "
        "Use numerical MCP tools to calculate numbers."
    ),
    "thermodynamics": (
        "State the system and sign convention. Delta U = Q - W_by: Q is positive into the system, "
        "W_by is positive for work by the system. Work on the system is -W_by. Internal energy is a "
        "state function; heat and work are energy transfers and depend on the path. For boundary work, "
        "W_by = integral(P_external dV); P Delta V requires constant external pressure. "
        "For an ideal gas PV = nRT, absolute temperatures must be in kelvin: T_K = T_C + 273.15. "
        "At constant ideal-gas temperature Delta U = 0; heat need not be zero. Adiabatic means Q = 0, "
        "not constant temperature. Constant volume means zero boundary work, not zero heat. "
        "For sensible heating with no phase change Q = mc Delta T. A temperature change has the same "
        "numerical value in K and degrees Celsius; absolute temperatures do not. During a phase change "
        "use latent heat rather than mc Delta T alone. Energy is conserved; heat flows spontaneously "
        "from hotter to colder bodies. An isolated system's entropy cannot decrease. "
        "A Carnot engine's maximum efficiency is 1 - T_c/T_h, with T_h > T_c > 0 in kelvin. "
        "Real engines cannot exceed that bound. Distinguish efficiency from refrigerator COP. "
        "Do not invent gas amounts, material properties, reservoir temperatures, or a process. "
        "Use numerical MCP tools to calculate numbers."
    ),
    "angular_motion": (
        "For constant angular acceleration: omega_f = omega_i + alpha t, "
        "Delta theta = omega_i t + (1/2) alpha t^2, omega_f^2 = omega_i^2 + 2 alpha Delta theta. "
        "Use radians for s = r theta, v_t = r omega, a_t = r alpha, and a_c = omega^2 r. "
        "Torque is the vector tau = r cross F, magnitude r F sin(phi). "
        "A force whose line of action passes through the axis gives zero torque. "
        "Use the right-hand rule and a consistent positive rotation direction; counterclockwise is commonly positive. "
        "For a fixed axis and constant moment of inertia, tau_net = I alpha. "
        "Moment of inertia depends on mass distribution and the selected axis, not just total mass. "
        "L = I omega for fixed-axis rotation. Total angular momentum is conserved when net external torque "
        "about that axis is zero. With conserved L, decreasing I increases omega. "
        "Rotational kinetic energy K_rot = (1/2) I omega^2 need not stay constant when I changes; internal work can change it. "
        "Rolling without slipping requires v_cm = R omega and includes translational and rotational kinetic energy. "
        "Torque units are N m, I has kg m^2, L has kg m^2/s. Use numerical MCP tools to calculate numbers."
    ),
}


def register_principles(mcp, domain):
    name = f"get_{domain}_principles"
    db_logger = DatabaseLogger(domain)

    @mcp.tool(name=name)
    @create_tool_wrapper(db_logger, name)
    async def get_principles() -> str:
        """Get conceptual physics principles for this domain without hypothetical numerical inputs."""
        return PRINCIPLES[domain]
