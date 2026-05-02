import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# =============================================================================
# Base Values (Per-Unit System)
# =============================================================================
V_base  = 5
I_base  = 5
f_base  = 1e7
T_base  = 1 / f_base
k       = I_base / V_base

# =============================================================================
# Component Values
# =============================================================================
Vin = 15             # V
R2  = 1           # Ω
L1  = 22e-6          # H
L2  = 22e-6          # H
C1  = 44e-6        # F
C2  = 176e-6        # F
# Fixed-point fractional bits
N_FRAC = 24

# =============================================================================
# Simulation Parameters
# =============================================================================
f_sw     = 20e3
T_sw     = 1 / f_sw
D        = 0.4
N_cycles = 200
N_steps  = int(N_cycles * T_sw / T_base)
t_end    = N_cycles * T_sw

# =============================================================================
# Derived Scaling Coefficients
# =============================================================================
aL1 = T_base / L1
aL2 = T_base / L2
aC1 = T_base / C1
aC2 = T_base / C2

# =============================================================================
# Utility
# =============================================================================
def quantize(mat, n_bits):
    m = np.array(mat, dtype=float)
    scale = 2 ** n_bits
    return np.round(m * scale) / scale

def pwm(t, duty, t_period):
    return (t % t_period) < duty * t_period

def pole_report(name, A):
    mag = np.abs(np.linalg.eigvals(A))
    status = "STABLE" if np.all(mag < 1) else ("MARGINAL" if np.all(np.isclose(mag,1,atol=1e-6)) else "UNSTABLE")
    print(f"{name}: |λ| = {np.round(mag,6)} → {status}")

# =============================================================================
# Exact Discretization
# =============================================================================
def exact_discretize(A, B):
    I = np.eye(A.shape[0])
    Ad = expm(A)

    try:
        Bd = np.linalg.solve(A, (Ad - I)) @ B
    except np.linalg.LinAlgError:
        Bd = (I + 0.5*A + (1/6)*A@A) @ B

    return Ad, Bd.flatten()

# =============================================================================
# Ćuk State Matrices (continuous, per-unit)
# =============================================================================
Aon = np.array([
    [      0,       0,          0,           0    ],  # iL1: no state coupling
    [      0,       0,       aL2,         -aL2    ],  # iL2: vC1-vC2 / L2
    [      0,  -k*aC1,         0,           0    ],  # vC1: -iL2/C1
    [      0,   k*aC2,         0,  -aC2 / R2      ],  # vC2: (iL2 - vC2/R)/C2
])

Aoff = np.array([
    [      0,       0,      -aL1,           0    ],  # iL1: (Vin-vC1)/L1  (Vin via B)
    [      0,       0,         0,        -aL2    ],  # iL2: -vC2/L2
    [  k*aC1,       0,         0,           0    ],  # vC1: iL1/C1
    [      0,  k*aC2,          0,  -aC2 / R2      ],  # vC2: (iL2 - vC2/R)/C2
])

Bon = np.array([[ aL1 ],   # iL1: Vin/L1
                [   0 ],
                [   0 ],
                [   0 ],])
Boff = Bon

# =============================================================================
# Exact Discrete Matrices (computed in full precision)
# =============================================================================
EX_M_on,  EX_v_on  = exact_discretize(Aon,  Bon)
EX_M_off, EX_v_off = exact_discretize(Aoff, Boff)

# Optional: quantize AFTER discretization (important!)
EX_M_on  = quantize(EX_M_on,  N_FRAC)
EX_v_on  = quantize(EX_v_on,  N_FRAC)
EX_M_off = quantize(EX_M_off, N_FRAC)
EX_v_off = quantize(EX_v_off, N_FRAC)

# =============================================================================
# Pole Check (this should now be correct)
# =============================================================================
print("Exact discretization poles:")
pole_report("ON ", EX_M_on)
pole_report("OFF", EX_M_off)
print()

# =============================================================================
# Simulation Loop
# =============================================================================
u_in    = Vin / V_base
t_arr   = np.linspace(0, t_end, N_steps, endpoint=False)
vR_EX   = np.empty(N_steps)

x = np.zeros(4)

for i, t in enumerate(t_arr):
    vR_EX[i] = x[3] * V_base

    if pwm(t, D, T_sw):
        x = EX_M_on  @ x + EX_v_on  * u_in
    else:
        x = EX_M_off @ x + EX_v_off * u_in

# =============================================================================
# Ideal Output
# =============================================================================
Vout_ideal = (D / (1 - D)) * Vin

# =============================================================================
# Plot
# =============================================================================
t_us = t_arr * 1e6

plt.figure(figsize=(12,5))
plt.plot(t_us, vR_EX, linewidth=0.8, label="Exact Discretization")
plt.axhline(Vout_ideal, linestyle='--', label=f'Ideal = {Vout_ideal:.2f} V')

plt.title("Ćuk Converter — Exact Discretization (Stable)")
plt.xlabel("Time (µs)")
plt.ylabel("Load Voltage (V)")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

print(f"Steady-state V_out (ideal): {Vout_ideal:.4f} V")
print(f"Simulated final V_out:     {vR_EX[-100:].mean():.4f} V")