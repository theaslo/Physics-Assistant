"""Render core equations from executed domain calculations, never from prompt keywords."""

import json
import re

NUMBER = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


def _value(text, pattern):
    matches = re.findall(pattern.format(n=NUMBER), text, re.M)
    return matches[-1] if matches else None


def _data(arguments, key):
    value = arguments[key]
    return json.loads(value) if isinstance(value, str) else value


def ground_domain_working(worked, outputs):
    """Keep the explanation, while anchoring the displayed algebra to actual tool output."""
    steps = []
    outcomes = []
    for output in outputs:
        name, args, text = output["tool"], output["arguments"], output["result"]
        step = None
        try:
            if name == "calculate_kinetic_energy_tool":
                result = _value(text, r"^\s*KE = ({n}) J")
                if result:
                    step = (r"K = \frac12 mv^2",
                            rf"K = \frac12({args['mass']:g}\,\mathrm{{kg}})({args['velocity']:g}\,\mathrm{{m/s}})^2 = {result}\,\mathrm{{J}}",
                            "Kinetic energy is nonnegative and does not depend on the sign of velocity.")
            elif name == "calculate_gravitational_potential_energy_tool":
                result = _value(text, r"^\s*PE = ({n}) J")
                if result:
                    step = (r"U_g = mgh",
                            rf"U_g = ({args['mass']:g})({args.get('gravity', 9.81):g})({args['height']:g}) = {result}\,\mathrm{{J}}",
                            "Potential energy uses the chosen zero height; changes depend on the difference between endpoint heights.")
            elif name == "calculate_elastic_potential_energy_tool":
                result = _value(text, r"^\s*PE = ({n}) J")
                if result:
                    step = (r"U_s = \frac12 kx^2",
                            rf"U_s = \frac12({args['spring_constant']:g})({args['displacement']:g})^2 = {result}\,\mathrm{{J}}",
                            "Stretching and compressing by the same magnitude store the same nonnegative spring energy.")
            elif name == "calculate_work_tool":
                result = _value(text, r"^\s*W = ({n}) J")
                if result:
                    step = (r"W = Fd\cos\theta",
                            rf"W = ({args['force']:g})({args['displacement']:g})\cos({args.get('angle_degrees', 0):g}^\circ) = {result}\,\mathrm{{J}}",
                            "The sign follows the force component along displacement; only net work equals the total change in kinetic energy.")
            elif name == "calculate_power_tool":
                result = _value(text, r"= ({n}) W\s*$")
                if result:
                    step = (r"P_{\rm avg}=\frac{W}{\Delta t}",
                            rf"P_{{\rm avg}}=\frac{{{args['work']:g}\,\mathrm{{J}}}}{{{args['time']:g}\,\mathrm{{s}}}}={result}\,\mathrm{{W}}",
                            "A watt is a joule per second; the elapsed time must be positive.")
            elif name == "work_energy_theorem":
                data = _data(args, "problem_data")
                m, vi = data["mass"], data["initial_velocity"]
                if "final_velocity" in data:
                    result = _value(text, r"^\s*Net work done: W_net = .* = ({n}) J")
                    if result:
                        step = (r"W_{\rm net}=K_f-K_i=\frac12m(v_f^2-v_i^2)",
                                rf"W_{{\rm net}}=\frac12({m:g})[({data['final_velocity']:g})^2-({vi:g})^2]={result}\,\mathrm{{J}}",
                                "Net work equals the change in kinetic energy; negative net work reduces the speed.")
                else:
                    result = _value(text, r"^\s*Final velocity: v_f = .* = ({n}) m/s")
                    work = data.get("work_done")
                    if work is None:
                        work = _value(text, r"^\s*Work by force: W = .* = ({n}) J")
                    if result and work is not None:
                        radicand = rf"({vi:g})^2+\frac{{2({work})}}{{{m:g}}}"
                        step = (r"\frac12mv_f^2=\frac12mv_i^2+W_{\rm net}",
                                rf"v_f=\sqrt{{{radicand}}}={result}\,\mathrm{{m/s}}",
                                "The square root determines speed. Work-energy alone does not determine the direction; the kinetic energy cannot be negative.")
            elif name == "energy_with_friction":
                data = _data(args, "friction_data")
                result = _value(text, r"^\s*v_f = .* = ({n}) m/s")
                work = _value(text, r"^\s*Work by friction: W_friction = ({n}) J")
                if result and work is not None:
                    m, vi = data["mass"], data["initial_velocity"]
                    g, hi, hf = data.get("gravity", 9.81), data.get("initial_height", 0), data.get("final_height", 0)
                    friction = rf"W_f={work}\,\mathrm{{J}}"
                    if "friction_coefficient" in data and "distance" in data:
                        friction = rf"W_f=-\mu_k mgd=-({data['friction_coefficient']:g})({m:g})({g:g})({data['distance']:g})={work}\,\mathrm{{J}}"
                    radicand = rf"({vi:g})^2+2({g:g})({hi:g}-{hf:g})+\frac{{2({work})}}{{{m:g}}}"
                    step = (r"K_i+U_i+W_f=K_f+U_f",
                            r"\begin{gathered}" + friction + r"\\"
                            + rf"v_f=\sqrt{{{radicand}}}={result}\,\mathrm{{m/s}}\end{{gathered}}",
                            "Sliding friction removes mechanical energy and transfers it to internal energy. Total energy is conserved; mechanical energy need not be.")
            elif name == "energy_conservation":
                data = _data(args, "system_data")
                result = _value(text, r"^\s*(?:Final speed: )?v_f = ({n}) m/s")
                if result and {"initial_height", "final_height", "initial_velocity"} <= data.keys():
                    hi, hf, vi, g = data["initial_height"], data["final_height"], data["initial_velocity"], data.get("gravity", 9.81)
                    step = (r"K_i+U_i=K_f+U_f,\qquad v_f^2=v_i^2+2g(h_i-h_f)",
                            rf"v_f=\sqrt{{({vi:g})^2+2({g:g})({hi:g}-{hf:g})}}={result}\,\mathrm{{m/s}}",
                            "This assumes zero nonconservative work. Mass cancels from the speed relation, and changing the potential-energy zero does not change the height difference.")
            elif name == "angular_kinematics":
                data = _data(args, "kinematics_data")
                # All values below come from the completed analysis, including
                # any acceleration or time solved by the calculator.
                w0 = _value(text, r"omega_0 = ({n}) rad/s")
                a = _value(text, r"alpha = ({n}) rad/s\^2")
                t = _value(text, r"t = ({n}) s")
                wf = _value(text, r"^omega_f = .* = ({n}) rad/s")
                theta = _value(text, r"^theta_f = .* = ({n}) rad")
                if all(value is not None for value in (w0, a, t, wf, theta)):
                    step = (r"\omega_f=\omega_i+\alpha t,\qquad \theta_f=\theta_i+\omega_i t+\frac12\alpha t^2",
                            rf"\begin{{gathered}}\omega_f={w0}+({a})({t})={wf}\,\mathrm{{rad/s}}\\"
                            rf"\theta_f={data.get('theta_0', 0):g}+({w0})({t})+\frac12({a})({t})^2={theta}\,\mathrm{{rad}}\end{{gathered}}",
                            "These relations require constant angular acceleration. Signed angular velocities use the same chosen positive direction.")
            elif name == "rotational_energy":
                data = _data(args, "energy_data")
                result = _value(text, r"^\s*KE_rot = .* = ({n}) J")
                if result and "moment_of_inertia" in data:
                    step = (r"K_{\rm rot}=\frac12 I\omega^2",
                            rf"K_{{\rm rot}}=\frac12({data['moment_of_inertia']:g})({data['angular_velocity']:g})^2={result}\,\mathrm{{J}}",
                            "Rotational kinetic energy is nonnegative. Inertia must refer to the stated axis.")
            elif name == "angular_momentum_conservation":
                data = _data(args, "momentum_data")
                if "figure_skater" in data:
                    skater = data["figure_skater"]
                    i1, w1, i2 = (skater[k] for k in ("I_extended", "omega_extended", "I_tucked"))
                else:
                    i1, w1, i2 = data["initial"]["I"], data["initial"]["omega"], data["final"]["I"]
                result = _value(text, r"^\s*ω₂ = .* = ({n}) rad/s")
                if result:
                    step = (r"I_i\omega_i=I_f\omega_f",
                            rf"\omega_f=\frac{{I_i\omega_i}}{{I_f}}=\frac{{({i1:g})({w1:g})}}{{{i2:g}}}={result}\,\mathrm{{rad/s}}",
                            "Angular momentum is conserved only with negligible net external torque about the chosen axis; rotational kinetic energy can change through internal work.")
                    ki = _value(text, r"Initial kinetic energy: KE₁ = .* = ({n}) J")
                    kf = _value(text, r"Final kinetic energy: KE₂ = .* = ({n}) J")
                    change = _value(text, r"Energy change: ΔKE = KE₂ - KE₁ = ({n}) J")
                    if all(value is not None for value in (ki, kf, change)):
                        comparison = ("Rotational kinetic energy is not conserved; internal work accounts for the change."
                                      if float(change) != 0 else "Rotational kinetic energy is unchanged at the shown precision.")
                        steps.append(step)
                        step = (r"K_i=\frac12 I_i\omega_i^2,\qquad K_f=\frac12 I_f\omega_f^2",
                                      rf"\begin{{gathered}}K_i=\frac12({i1:g})({w1:g})^2={ki}\,\mathrm{{J}}\\"
                                      rf"K_f=\frac12({i2:g})({result})^2={kf}\,\mathrm{{J}}\\"
                                      rf"\Delta K={kf}-{ki}={change}\,\mathrm{{J}}\end{{gathered}}", comparison)
                        outcomes.append(f"Rotational kinetic energy changes from {ki} J to {kf} J (change {change} J). " + comparison)
            elif name == "heat_transfer":
                data = _data(args, "heat_data")
                q, m, c, dt = [next((data[k] for k in keys if data.get(k) is not None), None)
                               for keys in (("heat", "Q"), ("mass", "m"), ("specific_heat", "c"),
                                            ("temp_change", "delta_T", "dT"))]
                if q is None:
                    result = _value(text, r"^Q = ({n}) J")
                    substitution = rf"Q=({m:g})({c:g})({dt:g})={result}\,\mathrm{{J}}"
                elif m is None:
                    result = _value(text, r"^m = ({n}) kg")
                    substitution = rf"m=\frac{{Q}}{{c\Delta T}}=\frac{{{q:g}}}{{({c:g})({dt:g})}}={result}\,\mathrm{{kg}}"
                elif c is None:
                    result = _value(text, r"^c = ({n}) J")
                    substitution = rf"c=\frac{{Q}}{{m\Delta T}}=\frac{{{q:g}}}{{({m:g})({dt:g})}}={result}\,\mathrm{{J/(kg\,K)}}"
                elif dt is None:
                    result = _value(text, r"^ΔT = ({n}) K")
                    substitution = rf"\Delta T=\frac{{Q}}{{mc}}=\frac{{{q:g}}}{{({m:g})({c:g})}}={result}\,\mathrm{{K}}"
                else:
                    result = None
                if result:
                    step = (r"Q=mc\Delta T", substitution,
                            "This relation assumes sensible heating with no phase change. Cooling gives negative heat into the object; a temperature difference has the same numerical value in K and degrees Celsius.")
            elif name == "ideal_gas_law":
                data = _data(args, "gas_data")
                p, v, n, t = [next((data[k] for k in keys if data.get(k) is not None), None)
                              for keys in (("pressure", "P"), ("volume", "V"), ("moles", "n"), ("temperature", "T"))]
                if p is None:
                    result = _value(text, r"^P = ({n}) Pa")
                    substitution = rf"P=\frac{{nRT}}{{V}}=\frac{{({n:g})(8.314)({t:g})}}{{{v:g}}}={result}\,\mathrm{{Pa}}"
                elif v is None:
                    result = _value(text, r"^V = ({n}) m")
                    substitution = rf"V=\frac{{nRT}}{{P}}=\frac{{({n:g})(8.314)({t:g})}}{{{p:g}}}={result}\,\mathrm{{m^3}}"
                elif n is None:
                    result = _value(text, r"^n = ({n}) mol")
                    substitution = rf"n=\frac{{PV}}{{RT}}=\frac{{({p:g})({v:g})}}{{(8.314)({t:g})}}={result}\,\mathrm{{mol}}"
                elif t is None:
                    result = _value(text, r"^T = ({n}) K")
                    substitution = rf"T=\frac{{PV}}{{nR}}=\frac{{({p:g})({v:g})}}{{({n:g})(8.314)}}={result}\,\mathrm{{K}}"
                else:
                    result = None
                if result:
                    step = (r"PV=nRT", substitution,
                            "Absolute temperature is in kelvin and the ideal-gas approximation is assumed; pressure, volume and gas amount are positive.")
            elif name == "first_law_thermodynamics":
                q = _value(text, r"Q = ({n}) J")
                w = _value(text, r"W_by = ({n}) J")
                du = _value(text, r"^ΔU = .* = ({n}) J")
                if all(value is not None for value in (q, w, du)):
                    step = (r"\Delta U=Q-W_{\rm by}",
                            rf"\Delta U=({q}\,\mathrm{{J}})-({w}\,\mathrm{{J}})={du}\,\mathrm{{J}}",
                            "Heat into the system is positive; work by the system removes energy. Compression work by the system is negative.")
            elif name == "constant_pressure_work":
                data = _data(args, "process_data")
                result = _value(text, r"^W_by = .* = ({n}) J")
                if result:
                    step = (r"W_{\rm by}=P(V_f-V_i)",
                            rf"W_{{\rm by}}=({data['pressure']:g})({data['final_volume']:g}-{data['initial_volume']:g})={result}\,\mathrm{{J}}",
                            "This formula assumes constant pressure. Expansion gives positive work by the gas; compression gives negative work.")
            elif name == "carnot_efficiency":
                data = _data(args, "carnot_data")
                result = _value(text, r"^η = ({n}) =")
                hot, cold = data.get("T_hot", data.get("Th")), data.get("T_cold", data.get("Tc"))
                if result:
                    step = (r"\eta_{\max}=1-\frac{T_c}{T_h}",
                            rf"\eta_{{\max}}=1-\frac{{{cold:g}\,\mathrm{{K}}}}{{{hot:g}\,\mathrm{{K}}}}={result}",
                            "Both reservoir temperatures are absolute kelvin values. This is an upper bound, with efficiency between zero and one.")
        except (KeyError, TypeError, ValueError):
            # Unrecognized composite calculations retain their complete,
            # model-written working based on the verified tool results.
            continue
        if step:
            steps.append(step)
    if not steps:
        return worked
    result = worked.get("result", "")
    for outcome in outcomes:
        if outcome not in result:
            result += "\n\n" + outcome
    return {**worked,
            "result": result,
            "equation": r"\begin{gathered}" + r" \\ ".join(s[0] for s in steps) + r"\end{gathered}",
            "substitution": r"\begin{gathered}" + r" \\ ".join(s[1] for s in steps) + r"\end{gathered}",
            "check": " ".join(dict.fromkeys(s[2] for s in steps))}
