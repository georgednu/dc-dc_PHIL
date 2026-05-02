"""
Buck Converter Simulator — Per-Unit State Space
================================================
Base values:
    V_base = 5 V
    I_base = 5 A
    R_base = 1 Ω
    P_base = 25 W
    f_sw   = 2 MHz  →  ω_base = 2π·2e6 rad/s
    L_base = R_base / ω_base = 79.58 nH
    C_base = 1 / (R_base · ω_base) = 79.58 nF

State variables (per-unit):
    x1 = iL_pu  (inductor current)
    x2 = vC_pu  (capacitor voltage = output voltage)

State space equations (continuous, per-unit):
  ON  phase (switch closed):  dx/dt = A_on  · x + B_on  · Vin_pu
  OFF phase (switch open):    dx/dt = A_off · x + B_off · Vin_pu

where (normalized time uses ω_base as implicit scale — but here we use
actual time in seconds with pu state variables, so L and C appear as
time constants L_pu/ω_base etc. — see derivation in comments below):

Actual inductor equation:  L · diL/dt = Vin - vC          (ON)
                           L · diL/dt =    - vC            (OFF)
Actual capacitor equation: C · dvC/dt = iL - vC/R

Normalizing (divide voltage eq by V_base, current eq by I_base):
  L · I_base · d(iL_pu)/dt = V_base·(Vin_pu - vC_pu)
  → d(iL_pu)/dt = (V_base / (L · I_base)) · (Vin_pu - vC_pu)
                = (R_base / L) · (Vin_pu - vC_pu)       [since V_base/I_base = R_base]
                = (1/L_pu_time) · (Vin_pu - vC_pu)      [L_pu_time = L/R_base, seconds]

  C · V_base · d(vC_pu)/dt = I_base·(iL_pu - vC_pu·V_base/R)
  → d(vC_pu)/dt = (I_base / (C · V_base)) · (iL_pu - vC_pu/R_pu)
                = (R_base / (R·C)) ... simplifies to:
                = (1/C_pu_time) · (iL_pu - vC_pu/R_pu)  [C_pu_time = C·R_base, seconds]

So the state matrices in actual time but pu state variables are:
  A_on  = [[ 0,        -1/τL ],     B_on  = [1/τL]
            [ 1/τC,  -1/(τC·R_pu)]]           [0  ]

  A_off = [[ 0,        -1/τL ],     B_off = [0]
            [ 1/τC,  -1/(τC·R_pu)]]           [0]

NOTE: A_on == A_off. The only difference between ON and OFF phases is
the input term B·Vin_pu. When ON, the source drives the inductor; when
OFF, the inductor freewheels through the diode with no source voltage.

where τL = L/R_base  (seconds),  τC = C*R_base  (seconds),  R_pu = R/R_base
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
#  BASE VALUES
# ─────────────────────────────────────────────
V_base = 5.0        # V
I_base = 5.0        # A
R_base = 1.0        # Ω  (= V_base / I_base)
P_base = 25.0       # W  (= V_base * I_base)
f_sw   = 2e6        # Hz
omega_base = 2 * np.pi * f_sw
L_base = R_base / omega_base          # 79.58 nH
C_base = 1.0 / (R_base * omega_base)  # 79.58 nF

print("=" * 50)
print("BASE VALUES")
print("=" * 50)
print(f"  V_base    = {V_base} V")
print(f"  I_base    = {I_base} A")
print(f"  R_base    = {R_base} Ω")
print(f"  P_base    = {P_base} W")
print(f"  f_sw      = {f_sw/1e6:.1f} MHz")
print(f"  ω_base    = {omega_base:.4e} rad/s")
print(f"  L_base    = {L_base*1e9:.4f} nH")
print(f"  C_base    = {C_base*1e9:.4f} nF")
print()

# ─────────────────────────────────────────────
#  CIRCUIT PARAMETERS  ← experiment here
# ─────────────────────────────────────────────
Vin_actual  = 5.0       # V   input voltage
L_actual    = 500e-9    # H   inductor  (500 nH)
C_actual    = 470e-9    # F   capacitor (470 nF)
R_actual    = 2.0       # Ω   load resistance
D           = 0.5       # duty cycle (0 < D < 1)

# Simulation time
n_cycles    = 100       # number of switching cycles to simulate
steps_per_cycle = 100  # time steps per cycle

# ─────────────────────────────────────────────
#  CONVERT PARAMETERS TO PER-UNIT
# ─────────────────────────────────────────────
Vin_pu  = Vin_actual / V_base
L_pu    = L_actual   / L_base       # dimensionless pu value
C_pu    = C_actual   / C_base
R_pu    = R_actual   / R_base

# Time constants used in state equations (actual seconds)
tauL = L_actual / R_base            # = L_pu / omega_base
tauC = C_actual * R_base            # = C_pu / omega_base

print("=" * 50)
print("CIRCUIT PARAMETERS")
print("=" * 50)
print(f"  Vin   = {Vin_actual} V       → {Vin_pu:.4f} pu")
print(f"  L     = {L_actual*1e9:.1f} nH      → {L_pu:.4f} pu")
print(f"  C     = {C_actual*1e9:.1f} nF    → {C_pu:.4f} pu")
print(f"  R     = {R_actual} Ω        → {R_pu:.4f} pu")
print(f"  D     = {D}")
print(f"  τL    = {tauL*1e9:.2f} ns")
print(f"  τC    = {tauC*1e9:.2f} ns")
print()

# Theoretical steady-state (ideal buck)
Vout_theory = D * Vin_actual
Iout_theory = Vout_theory / R_actual
print(f"  Theoretical Vout (ideal) = D·Vin = {Vout_theory:.4f} V")
print(f"  Theoretical Iout (ideal) = Vout/R = {Iout_theory:.4f} A")
print()

# ─────────────────────────────────────────────
#  STATE SPACE MATRICES (pu state, actual time)
# ─────────────────────────────────────────────
#   state x = [iL_pu, vC_pu]^T

A_on = np.array([
    [ 0,               -1/tauL         ],
    [ 1/tauC,  -1/(tauC * R_pu)        ]
])

A_off = np.array([
    [ 0,               -1/tauL         ],
    [ 1/tauC,  -1/(tauC * R_pu)        ]
])

B_on  = np.array([1/tauL, 0.0])
B_off = np.array([0.0,    0.0])

# ─────────────────────────────────────────────
#  NUMERICAL INTEGRATION (Forward Euler)
# ─────────────────────────────────────────────
T_sw   = 1.0 / f_sw
dt     = T_sw / steps_per_cycle
t_on   = D * T_sw
t_off  = (1 - D) * T_sw
steps_on  = max(1, int(round(t_on  / dt)))
steps_off = max(1, int(round(t_off / dt)))

total_steps = n_cycles * (steps_on + steps_off)

# Storage
time_arr  = np.zeros(total_steps)
iL_pu_arr = np.zeros(total_steps)
vC_pu_arr = np.zeros(total_steps)

# Initial conditions (pu)
x = np.array([0.0, 0.0])   # [iL_pu, vC_pu]

idx = 0
t   = 0.0

for _ in range(n_cycles):
    # ON phase
    for _ in range(steps_on):
        time_arr[idx]  = t
        iL_pu_arr[idx] = x[0]
        vC_pu_arr[idx] = x[1]
        dxdt = A_on @ x + B_on * Vin_pu
        x = x + dt * dxdt
        t += dt
        idx += 1

    # OFF phase
    for _ in range(steps_off):
        time_arr[idx]  = t
        iL_pu_arr[idx] = x[0]
        vC_pu_arr[idx] = x[1]
        dxdt = A_off @ x + B_off * Vin_pu
        x = x + dt * dxdt
        t += dt
        idx += 1

# ─────────────────────────────────────────────
#  CONVERT BACK TO ACTUAL VALUES
# ─────────────────────────────────────────────
iL_actual_arr = iL_pu_arr * I_base
vC_actual_arr = vC_pu_arr * V_base

# Steady-state averages (last 10 cycles)
ss_start = total_steps - 10 * (steps_on + steps_off)
Vout_ss = np.mean(vC_actual_arr[ss_start:])
Iout_ss = np.mean(iL_actual_arr[ss_start:])
ripple_V = np.max(vC_actual_arr[ss_start:]) - np.min(vC_actual_arr[ss_start:])
ripple_I = np.max(iL_actual_arr[ss_start:]) - np.min(iL_actual_arr[ss_start:])

print("=" * 50)
print("SIMULATION RESULTS (steady-state)")
print("=" * 50)
print(f"  Vout (avg)     = {Vout_ss:.4f} V  ({Vout_ss/V_base:.4f} pu)")
print(f"  Iout (avg)     = {Iout_ss:.4f} A  ({Iout_ss/I_base:.4f} pu)")
print(f"  Voltage ripple = {ripple_V*1000:.4f} mV  ({ripple_V/V_base*100:.4f} %)")
print(f"  Current ripple = {ripple_I*1000:.4f} mA  ({ripple_I/I_base*100:.4f} %)")
print()

# ─────────────────────────────────────────────
#  PLOT
# ─────────────────────────────────────────────
# Full arrays for transient view
t_us       = time_arr       * 1e6   # µs
vC_full    = vC_actual_arr
iL_full    = iL_actual_arr

# Steady-state zoom: last 20 cycles
ss_idx = total_steps - 20 * (steps_on + steps_off)
t_ss   = time_arr[ss_idx:] * 1e6
vC_ss  = vC_actual_arr[ss_idx:]
iL_ss  = iL_actual_arr[ss_idx:]

# Per-cycle envelope (peak/min/avg) for smooth transient overlay
cycle_steps = steps_on + steps_off
env_t    = np.zeros(n_cycles)
env_vmax = np.zeros(n_cycles)
env_vmin = np.zeros(n_cycles)
env_vavg = np.zeros(n_cycles)
env_imax = np.zeros(n_cycles)
env_imin = np.zeros(n_cycles)
env_iavg = np.zeros(n_cycles)
for k in range(n_cycles):
    sl = slice(k * cycle_steps, (k + 1) * cycle_steps)
    env_t[k]    = time_arr[k * cycle_steps] * 1e6
    env_vmax[k] = np.max(vC_full[sl])
    env_vmin[k] = np.min(vC_full[sl])
    env_vavg[k] = np.mean(vC_full[sl])
    env_imax[k] = np.max(iL_full[sl])
    env_imin[k] = np.min(iL_full[sl])
    env_iavg[k] = np.mean(iL_full[sl])

def style_ax(ax, title):
    ax.set_title(title, color='#e0e0e0', fontsize=9)
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='#aaaaaa', labelsize=7)
    for sp in ax.spines.values():
        sp.set_color('#333333')
    ax.xaxis.label.set_color('#aaaaaa')
    ax.yaxis.label.set_color('#aaaaaa')

fig = plt.figure(figsize=(14, 10), facecolor='#0f0f0f')
fig.suptitle(
    f"Buck Converter Transient  —  Vin={Vin_actual}V  D={D}  "
    f"L={L_actual*1e9:.0f}nH  C={C_actual*1e9:.0f}nF  R={R_actual}Ω  fsw={f_sw/1e6:.0f}MHz",
    color='#e0e0e0', fontsize=11, y=0.99
)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.35)

# ── Row 0: Full transient envelopes ──────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.fill_between(env_t, env_vmin, env_vmax, color='#00d4aa', alpha=0.25, label='Ripple band')
ax1.plot(env_t, env_vavg, color='#00d4aa', linewidth=1.2, label='Cycle avg')
ax1.axhline(Vout_theory, color='#ff6b6b', linewidth=1.0, linestyle='--', label=f'D·Vin = {Vout_theory:.3f}V')
ax1.set_xlabel('Time (µs)', fontsize=8)
ax1.set_ylabel('Voltage (V)', fontsize=8)
ax1.legend(fontsize=7, facecolor='#1a1a1a', labelcolor='#e0e0e0')
style_ax(ax1, 'Output Voltage — Full Transient (actual)')

ax2 = fig.add_subplot(gs[0, 1])
ax2.fill_between(env_t, env_imin, env_imax, color='#4fc3f7', alpha=0.25, label='Ripple band')
ax2.plot(env_t, env_iavg, color='#4fc3f7', linewidth=1.2, label='Cycle avg')
ax2.axhline(Vout_theory / R_actual, color='#ff6b6b', linewidth=1.0, linestyle='--',
            label=f'Ideal IL = {Vout_theory/R_actual:.3f}A')
ax2.set_xlabel('Time (µs)', fontsize=8)
ax2.set_ylabel('Current (A)', fontsize=8)
ax2.legend(fontsize=7, facecolor='#1a1a1a', labelcolor='#e0e0e0')
style_ax(ax2, 'Inductor Current — Full Transient (actual)')

# ── Row 1: Steady-state ripple zoom ──────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(t_ss, vC_ss, color='#00d4aa', linewidth=0.9)
ax3.axhline(Vout_ss, color='#ffd166', linewidth=1.0, linestyle=':', label=f'Avg = {Vout_ss:.4f}V')
ax3.set_xlabel('Time (µs)', fontsize=8)
ax3.set_ylabel('Voltage (V)', fontsize=8)
ax3.legend(fontsize=7, facecolor='#1a1a1a', labelcolor='#e0e0e0')
style_ax(ax3, 'Output Voltage — Steady-State Ripple (actual)')

ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(t_ss, iL_ss, color='#4fc3f7', linewidth=0.9)
ax4.axhline(Iout_ss, color='#ffd166', linewidth=1.0, linestyle=':', label=f'Avg = {Iout_ss:.4f}A')
ax4.set_xlabel('Time (µs)', fontsize=8)
ax4.set_ylabel('Current (A)', fontsize=8)
ax4.legend(fontsize=7, facecolor='#1a1a1a', labelcolor='#e0e0e0')
style_ax(ax4, 'Inductor Current — Steady-State Ripple (actual)')

# ── Row 2: Phase portrait + summary ──────────
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(vC_pu_arr, iL_pu_arr, color='#c084fc', linewidth=0.4, alpha=0.3, label='Transient')
ax5.plot(vC_ss / V_base, iL_ss / I_base, color='#f0abfc', linewidth=0.9, alpha=0.9, label='Steady-state')
ax5.set_xlabel('vC (pu)', fontsize=8)
ax5.set_ylabel('iL (pu)', fontsize=8)
ax5.legend(fontsize=7, facecolor='#1a1a1a', labelcolor='#e0e0e0')
style_ax(ax5, 'Phase Portrait (pu)  —  Transient → Steady-State')

ax6 = fig.add_subplot(gs[2, 1])
ax6.axis('off')
ax6.set_facecolor('#1a1a1a')

# Estimate 2% settling time
settled_cycle = n_cycles - 1
for k in range(n_cycles):
    if abs(env_vavg[k] - Vout_ss) / max(Vout_ss, 1e-9) < 0.02:
        settled_cycle = k
        break
t_settle = env_t[settled_cycle]

summary = (
    f"  BASE VALUES\n"
    f"  ─────────────────────────\n"
    f"  V_base   = {V_base} V\n"
    f"  I_base   = {I_base} A\n"
    f"  R_base   = {R_base} Ω\n"
    f"  L_base   = {L_base*1e9:.2f} nH\n"
    f"  C_base   = {C_base*1e9:.2f} nF\n\n"
    f"  TRANSIENT\n"
    f"  ─────────────────────────\n"
    f"  ~2% settle  ≈ {t_settle:.2f} µs\n"
    f"               ({settled_cycle} cycles)\n\n"
    f"  STEADY STATE\n"
    f"  ─────────────────────────\n"
    f"  Vout     = {Vout_ss:.4f} V  ({Vout_ss/V_base:.4f} pu)\n"
    f"  Iout     = {Iout_ss:.4f} A  ({Iout_ss/I_base:.4f} pu)\n"
    f"  ΔV       = {ripple_V*1000:.2f} mV  ({ripple_V/Vout_ss*100:.2f}%)\n"
    f"  ΔI       = {ripple_I*1000:.1f} mA  ({ripple_I/Iout_ss*100:.2f}%)"
)
ax6.text(0.05, 0.97, summary, transform=ax6.transAxes,
         fontsize=8, verticalalignment='top', fontfamily='monospace',
         color='#e0e0e0',
         bbox=dict(boxstyle='round', facecolor='#252525', alpha=0.8, edgecolor='#444'))

plt.savefig('./buck_sim_result.png', dpi=150, bbox_inches='tight',
            facecolor='#0f0f0f')
plt.close()
print("Plot saved to buck_sim_result.png")
