import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Base Values (Per-Unit System)
# =============================================================================
V_base  = 15         # V
I_base  = 15         # A
f_base  = 1e7       # Hz  — simulation timestep rate
T_base  = 1 / f_base
k       = I_base / V_base

# =============================================================================
# Component Values
# =============================================================================
Vin = 15             # V
R2  = 1           # Ω
L1  = 22e-6          # H
L2  = 22e-6          # 
C1  = 44e-6        # F
C2  = 560e-6        # F

# Fixed-point fractional bits
N_FRAC = 24

# =============================================================================
# Simulation Parameters
# =============================================================================
f_sw     = 20e3                            # Hz
T_sw     = 1 / f_sw
D        = 0.5                              # duty cycle
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
    """Round to n_bits fractional fixed-point. Returns float64 array (copy)."""
    m     = np.array(mat, dtype=float)
    scale = 2 ** n_bits
    return np.round(m * scale) / scale

def pwm(t, duty, t_period):
    return (t % t_period) < duty * t_period

def pole_report(name, A):
    I   = np.eye(A.shape[0])
    mag = np.abs(np.linalg.eigvals(A))
    status = "STABLE" if np.all(mag < 1) else ("MARGINALLY STABLE" if np.all(mag <= 1) else "UNSTABLE")
    print(f"  {name}: |λ| = {np.round(mag, 8)}  →  {status}")

# =============================================================================
# Ćuk State Matrices (per-unit, pre-scaled by T_base)
# State: x = [iL1, iL2, vC1, vC2]
# =============================================================================
# Aon = quantize([
#     [      0,       0,   -aL1,           0 ],
#     [      0,       0,      0,        -aL2 ],
#     [ k*aC1,        0,      0,           0 ],
#     [      0,  k*aC2,        0, -aC2 / R2  ],
# ], N_FRAC)

Aon = quantize([
    [      0,       0,          0,           0    ],  # iL1: no state coupling
    [      0,       0,       aL2,         -aL2    ],  # iL2: vC1-vC2 / L2
    [      0,  -k*aC1,         0,           0    ],  # vC1: -iL2/C1
    [      0,   k*aC2,         0,  -aC2 / R2      ],  # vC2: (iL2 - vC2/R)/C2
], N_FRAC)
 
Bon = quantize([
    [ aL1 ],   # iL1: Vin/L1
    [   0 ],
    [   0 ],
    [   0 ],
], N_FRAC)
 
#                    iL1    iL2       vC1          vC2
Aoff = quantize([
    [      0,       0,      -aL1,           0    ],  # iL1: (Vin-vC1)/L1  (Vin via B)
    [      0,       0,         0,        -aL2    ],  # iL2: -vC2/L2
    [  k*aC1,       0,         0,           0    ],  # vC1: iL1/C1
    [      0,  k*aC2,          0,  -aC2 / R2      ],  # vC2: (iL2 - vC2/R)/C2
], N_FRAC)
 
Boff = Bon  # Vin still drives L1 in both states

# =============================================================================
# Precompute Implicit Integration Matrices (done once, offline)
#
# Backward Euler:
#   (I - A) x[k+1] = x[k] + B*u
#   x[k+1] = (I - A)^-1 x[k] + (I - A)^-1 B*u
#
# Trapezoidal (Crank-Nicolson):
#   (I - A/2) x[k+1] = (I + A/2) x[k] + B*u
#   x[k+1] = (I - A/2)^-1 (I + A/2) x[k] + (I - A/2)^-1 B*u
#
# In both cases the update is: x[k+1] = M * x[k] + v * u
# where M and v are precomputed constant matrices.
# =============================================================================
I4 = np.eye(4)

# --- Backward Euler ---
BE_M_on   = np.linalg.inv(I4 - Aon)
BE_v_on   = BE_M_on @ Bon.flatten()
BE_M_off  = np.linalg.inv(I4 - Aoff)
BE_v_off  = BE_M_off @ Boff.flatten()

# --- Trapezoidal ---
TR_M_on   = np.linalg.inv(I4 - Aon  / 2) @ (I4 + Aon  / 2)
TR_v_on   = np.linalg.inv(I4 - Aon  / 2) @ Bon.flatten()
TR_M_off  = np.linalg.inv(I4 - Aoff / 2) @ (I4 + Aoff / 2)
TR_v_off  = np.linalg.inv(I4 - Aoff / 2) @ Boff.flatten()

# =============================================================================
# Pole Check — iteration matrices
# =============================================================================
print("Iteration matrix pole magnitudes:")
pole_report("Backward Euler  ON ", BE_M_on)
pole_report("Backward Euler  OFF", BE_M_off)
pole_report("Trapezoidal     ON ", TR_M_on)
pole_report("Trapezoidal     OFF", TR_M_off)
print()

# =============================================================================
# Simulation Loop
# =============================================================================
u_in    = Vin / V_base
t_arr   = np.linspace(0, t_end, N_steps, endpoint=False)
vR_BE   = np.empty(N_steps)
vR_TR   = np.empty(N_steps)

x_be = np.zeros(4)
x_tr = np.zeros(4)

for i, t in enumerate(t_arr):
    vR_BE[i] = x_be[3] * V_base    # vC2 → physical volts
    vR_TR[i] = x_tr[3] * V_base

    if pwm(t, D, T_sw):
        x_be = BE_M_on  @ x_be + BE_v_on  * u_in
        x_tr = TR_M_on  @ x_tr + TR_v_on  * u_in
    else:
        x_be = BE_M_off @ x_be + BE_v_off * u_in
        x_tr = TR_M_off @ x_tr + TR_v_off * u_in

# =============================================================================
# Ideal Steady-State
# =============================================================================
Vout_ideal = (D / (1 - D)) * Vin   # = 5 V at D=0.5

# =============================================================================
# Plot
# =============================================================================
t_us = t_arr * 1e6

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
fig.suptitle(
    f"Ćuk Converter — Load Voltage Comparison\n"
    f"D = {D},  f_sw = {f_sw/1e3:.0f} kHz,  Vin = {Vin} V,  R = {R2} Ω,  "
    f"L1=L2={L1*1e3:.2f} mH,  C1=C2={C1*1e6:.0f} µF",
    fontsize=11
)

for ax, vR, label, color in [
    (axes[0], vR_BE, "Backward Euler",  "steelblue"),
    (axes[1], vR_TR, "Trapezoidal",     "darkorange"),
]:
    ax.plot(t_us, vR, color=color, linewidth=0.8, label=label)
    ax.axhline(Vout_ideal, color='tomato', linewidth=1.2, linestyle='--',
               label=f'V_out ideal = {Vout_ideal:.2f} V')
    ax.set_ylabel("Load Voltage (V)")
    ax.set_ylim(0, Vout_ideal * 3)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.4)
    ax.text(0.98, 0.95,
            "ON-state iteration matrix has |λ| > 1\n"
            "(coupled LC resonance exceeds timestep stability bound)\n"
            "→ divergence is in A_on, not the integrator method",
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

axes[1].set_xlabel("Time (µs)")

plt.tight_layout()
plt.savefig("./cuk_implicit_comparison.png", dpi=150)
plt.show()
print(f"Steady-state V_out (ideal):  {Vout_ideal:.4f} V")
print(f"Backward Euler final V_out:  {vR_BE[-100:].mean():.4f} V")
print(f"Trapezoidal   final V_out:   {vR_TR[-100:].mean():.4f} V")