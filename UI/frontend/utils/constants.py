"""
Physics constants and conversion factors for the Physics Assistant UI
"""

# Fundamental Physical Constants
PHYSICS_CONSTANTS = {
    # Mechanics
    "g": 9.80665,  # Standard gravity (m/s²)
    "g_earth": 9.80665,  # Earth's surface gravity (m/s²)
    
    # Universal Constants
    "c": 2.99792458e8,  # Speed of light in vacuum (m/s)
    "G": 6.67430e-11,  # Gravitational constant (m³/kg⋅s²)
    "h": 6.62607015e-34,  # Planck constant (J⋅s)
    "hbar": 1.054571817e-34,  # Reduced Planck constant (J⋅s)
    
    # Electromagnetic Constants
    "e": 1.602176634e-19,  # Elementary charge (C)
    "epsilon_0": 8.8541878128e-12,  # Vacuum permittivity (F/m)
    "mu_0": 1.25663706212e-6,  # Vacuum permeability (H/m)
    "k_e": 8.9875517923e9,  # Coulomb constant (N⋅m²/C²)
    
    # Atomic and Molecular Constants
    "m_e": 9.1093837015e-31,  # Electron mass (kg)
    "m_p": 1.67262192369e-27,  # Proton mass (kg)
    "m_n": 1.67492749804e-27,  # Neutron mass (kg)
    "N_A": 6.02214076e23,  # Avogadro constant (mol⁻¹)
    "R": 8.314462618,  # Gas constant (J/mol⋅K)
    "k_B": 1.380649e-23,  # Boltzmann constant (J/K)
    
    # Other Useful Constants
    "sigma": 5.670374419e-8,  # Stefan-Boltzmann constant (W/m²⋅K⁴)
    "alpha": 7.2973525693e-3,  # Fine structure constant
    "a_0": 5.29177210903e-11,  # Bohr radius (m)
}

# Unit Conversion Factors
UNIT_CONVERSIONS = {
    # Length
    "length": {
        "m_to_cm": 100,
        "m_to_mm": 1000,
        "m_to_km": 0.001,
        "m_to_in": 39.3701,
        "m_to_ft": 3.28084,
        "m_to_yd": 1.09361,
        "m_to_mi": 0.000621371,
        "cm_to_m": 0.01,
        "mm_to_m": 0.001,
        "km_to_m": 1000,
        "in_to_m": 0.0254,
        "ft_to_m": 0.3048,
        "yd_to_m": 0.9144,
        "mi_to_m": 1609.34
    },
    
    # Mass
    "mass": {
        "kg_to_g": 1000,
        "kg_to_mg": 1e6,
        "kg_to_lb": 2.20462,
        "kg_to_oz": 35.274,
        "g_to_kg": 0.001,
        "mg_to_kg": 1e-6,
        "lb_to_kg": 0.453592,
        "oz_to_kg": 0.0283495
    },
    
    # Time
    "time": {
        "s_to_ms": 1000,
        "s_to_min": 1/60,
        "s_to_h": 1/3600,
        "s_to_day": 1/86400,
        "ms_to_s": 0.001,
        "min_to_s": 60,
        "h_to_s": 3600,
        "day_to_s": 86400
    },
    
    # Force
    "force": {
        "N_to_kN": 0.001,
        "N_to_dyne": 1e5,
        "N_to_lbf": 0.224809,
        "kN_to_N": 1000,
        "dyne_to_N": 1e-5,
        "lbf_to_N": 4.44822
    },
    
    # Energy
    "energy": {
        "J_to_kJ": 0.001,
        "J_to_cal": 0.239006,
        "J_to_kcal": 0.000239006,
        "J_to_eV": 6.242e18,
        "J_to_kWh": 2.77778e-7,
        "J_to_BTU": 0.000947817,
        "kJ_to_J": 1000,
        "cal_to_J": 4.184,
        "kcal_to_J": 4184,
        "eV_to_J": 1.602176634e-19,
        "kWh_to_J": 3.6e6,
        "BTU_to_J": 1055.06
    },
    
    # Power
    "power": {
        "W_to_kW": 0.001,
        "W_to_MW": 1e-6,
        "W_to_hp": 0.00134102,
        "kW_to_W": 1000,
        "MW_to_W": 1e6,
        "hp_to_W": 745.7
    },
    
    # Pressure
    "pressure": {
        "Pa_to_kPa": 0.001,
        "Pa_to_MPa": 1e-6,
        "Pa_to_atm": 9.86923e-6,
        "Pa_to_psi": 0.000145038,
        "Pa_to_torr": 0.00750062,
        "Pa_to_bar": 1e-5,
        "kPa_to_Pa": 1000,
        "MPa_to_Pa": 1e6,
        "atm_to_Pa": 101325,
        "psi_to_Pa": 6894.76,
        "torr_to_Pa": 133.322,
        "bar_to_Pa": 1e5
    },
    
    # Temperature (Note: These are for conversion, not addition/subtraction)
    "temperature": {
        "C_to_K_offset": 273.15,
        "F_to_C_scale": 5/9,
        "C_to_F_scale": 9/5,
        "F_to_C_offset": 32
    },
    
    # Angle
    "angle": {
        "rad_to_deg": 180 / 3.14159265359,
        "deg_to_rad": 3.14159265359 / 180,
        "rad_to_grad": 200 / 3.14159265359,
        "grad_to_rad": 3.14159265359 / 200
    }
}

# Common Physics Formulas (for reference)
PHYSICS_FORMULAS = {
    "kinematics": {
        "velocity": "v = u + at",
        "displacement": "s = ut + ½at²",
        "velocity_squared": "v² = u² + 2as",
        "average_velocity": "v_avg = (u + v) / 2"
    },
    
    "forces": {
        "newton_second": "F = ma",
        "weight": "W = mg",
        "friction": "f = μN",
        "hookes_law": "F = -kx"
    },
    
    "energy": {
        "kinetic": "KE = ½mv²",
        "potential_gravitational": "PE = mgh",
        "potential_elastic": "PE = ½kx²",
        "work": "W = F⋅d = Fd cos θ",
        "power": "P = W/t = F⋅v"
    },
    
    "momentum": {
        "momentum": "p = mv",
        "impulse": "J = Δp = FΔt",
        "conservation": "p_initial = p_final"
    },
    
    "rotation": {
        "angular_velocity": "ω = θ/t",
        "angular_acceleration": "α = ω/t",
        "torque": "τ = rF sin θ",
        "moment_of_inertia": "I = Σmr²",
        "rotational_kinetic": "KE_rot = ½Iω²",
        "angular_momentum": "L = Iω"
    },
    
    "waves": {
        "wave_speed": "v = fλ",
        "period": "T = 1/f",
        "simple_harmonic": "x = A cos(ωt + φ)"
    }
}

# Unit symbols and names
UNIT_SYMBOLS = {
    # Base SI units
    "m": "meter",
    "kg": "kilogram", 
    "s": "second",
    "A": "ampere",
    "K": "kelvin",
    "mol": "mole",
    "cd": "candela",
    
    # Derived SI units
    "N": "newton",
    "Pa": "pascal",
    "J": "joule",
    "W": "watt",
    "C": "coulomb",
    "V": "volt",
    "Ω": "ohm",
    "F": "farad",
    "H": "henry",
    "T": "tesla",
    "Wb": "weber",
    
    # Common non-SI units
    "°C": "degree Celsius",
    "°F": "degree Fahrenheit",
    "rad": "radian",
    "deg": "degree",
    "rpm": "revolutions per minute",
    "mph": "miles per hour",
    "kmh": "kilometers per hour"
}

# Physics topics and their associated keywords
PHYSICS_TOPICS = {
    "kinematics": [
        "position", "displacement", "velocity", "speed", "acceleration",
        "motion", "distance", "time", "graph", "slope"
    ],
    "forces": [
        "force", "newton", "mass", "acceleration", "friction", "tension",
        "normal", "weight", "gravity", "incline", "free body diagram"
    ],
    "energy": [
        "work", "energy", "kinetic", "potential", "conservation",
        "power", "efficiency", "joule", "calorie"
    ],
    "momentum": [
        "momentum", "impulse", "collision", "conservation", "elastic",
        "inelastic", "explosion", "center of mass"
    ],
    "rotation": [
        "rotation", "angular", "torque", "moment", "inertia", "axis",
        "spinning", "rolling", "gyroscope"
    ],
    "waves": [
        "wave", "frequency", "wavelength", "amplitude", "period",
        "oscillation", "vibration", "sound", "light"
    ],
    "electricity": [
        "charge", "current", "voltage", "resistance", "circuit",
        "ohm", "power", "battery", "capacitor"
    ],
    "magnetism": [
        "magnetic", "field", "flux", "induction", "coil",
        "generator", "motor", "transformer"
    ]
}

# Error tolerance for physics calculations
CALCULATION_TOLERANCE = {
    "default": 1e-6,
    "percentage": 0.01,  # 1% tolerance
    "significant_figures": 3
}