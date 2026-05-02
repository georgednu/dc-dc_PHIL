"""
Buck Converter — Piecewise Forward Euler (ON / OFF matrices)
=============================================================
Two separate state-space matrices for each switch state:

  SWITCH ON  (high-side on, low-side off):
    L·diL/dt = V_in - v_C      =>  diL/dt = (1/L)·V_in  - (1/L)·v_C
    C·dvC/dt = i_L  - v_C/R   =>  dvC/dt = (1/C)·i_L   - (1/RC)·v_C

  SWITCH OFF (high-side off, low-side on — synchronous, freewheeling):
    L·diL/dt = 0 - v_C         =>  diL/dt = -(1/L)·v_C
    C·dvC/dt = i_L - v_C/R    =>  dvC/dt = (1/C)·i_L  - (1/RC)·v_C

Both share identical A; only B differs (V_in drives inductor only when ON):

  A      = [  0,      -1/L   ]
           [ 1/C,   -1/(RC)  ]

  B_on   = [ 1/L ]      B_off = [ 0 ]
           [  0  ]               [ 0 ]

Forward Euler: x[k+1] = (I + A·dt)·x[k] + B·dt·u[k]
                       =  Ad·x[k]        + Bd·u[k]

  Ad is identical for ON and OFF (same A).
  Bd_on  = B_on  · dt  (non-zero only for i_L equation)
  Bd_off = [0, 0]^T    (no source during freewheeling)

Physical parameters
-------------------
  V_in  = 12 V,  D = 50%,  L = 10 uH,  C = 47 uF,  R = 1 Ohm
  f_sw  = 20 kHz  (control),  f_sim = 2 MHz (T_base = 500 ns)
  Steps per switching period: T_sw / dt = 50 us / 500 ns = 100

NOTE on operating point
-----------------------
  With L = 10 uH and f_sw = 20 kHz the theoretical peak-to-peak ripple is:
    DeltaiL = (Vin - Vout)*t_on / L = (12 - 6) * 25 us / 10 uH = 15 A
  The average load current is only 6 A, so the ripple is 250% of the average.
  This means i_L goes deeply negative each cycle (synchronous rectification
  allows reverse current). The AVERAGE values still converge exactly to
  i_L_avg = 6 A and v_C_avg = 6 V -- verified analytically below.
  A practical design would use L ~ 75 uH for ~30% ripple at 20 kHz.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ──────────────────────────────────────────────────────────────────────────────
# 1. Physical parameters
# ──────────────────────────────────────────────────────────────────────────────
V_in   = 12.0
D      = 0.50
L      = 100e-6
C      = 560e-6
R      = 1.0
f_sim  = 2e6
f_ctrl = 20e3
dt     = 1.0 / f_sim   # T_base = 500 ns

T_sw      = 1.0 / f_ctrl
steps_per = int(round(T_sw / dt))       # 100 steps per period
steps_on  = int(round(D * steps_per))   # 50 steps ON
steps_off = steps_per - steps_on        # 50 steps OFF

print("=" * 64)
print("Switching & simulation parameters")
print("=" * 64)
print(f"  f_ctrl           = {f_ctrl/1e3:.0f} kHz")
print(f"  f_sim  (T_base)  = {f_sim/1e6:.0f} MHz  ->  dt = {dt*1e9:.0f} ns")
print(f"  T_sw             = {T_sw*1e6:.1f} us")
print(f"  D                = {D:.0%}")
print(f"  Steps per period = {steps_per}  ({steps_on} ON / {steps_off} OFF)\n")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Continuous-time ON / OFF matrices (physical units)
# ──────────────────────────────────────────────────────────────────────────────
A_on  = np.array([[ 0.0,     -1.0/L      ],
                  [ 1.0/C,  -1.0/(R*C)   ]])

A_off = np.array([[ 0.0,     -1.0/L      ],
                  [ 1.0/C,  -1.0/(R*C)   ]])   # identical to A_on

B_on  = np.array([[1.0/L],
                  [0.0  ]])

B_off = np.array([[0.0],
                  [0.0]])

# Averaged-model steady state: A_avg*x_ss + B_avg*V_in = 0
B_avg = D * B_on + (1-D) * B_off
x_ss  = -np.linalg.solve(A_on, B_avg) * V_in
iL_ss, vC_ss = x_ss.flatten()

delta_iL_th = (V_in - vC_ss) * (D / f_ctrl) / L
delta_vC_th = delta_iL_th / (8.0 * C * f_ctrl)

print("=" * 64)
print("Continuous-time matrices (physical units)")
print("=" * 64)
print(f"A_on  = A_off =\n{A_on}\n")
print(f"B_on  =\n{B_on}\n")
print(f"B_off =\n{B_off}\n")
print(f"Averaged SS:  i_L = {iL_ss:.4f} A,  v_C = {vC_ss:.4f} V")
print(f"Expected:     D*Vin = {D*V_in:.3f} V\n")
print(f"Theoretical ripple:  DeltaiL = {delta_iL_th:.3f} A  ({delta_iL_th/iL_ss*100:.0f}% of avg)")
print(f"                     DeltavC = {delta_vC_th*1e3:.3f} mV\n")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Per-unit scaling
# ──────────────────────────────────────────────────────────────────────────────
V_base = 5.0
I_base = 5.0
T_base = dt
Z_base = V_base / I_base   # = 1 Ohm

S_x     = np.diag([1.0/I_base, 1.0/V_base])
S_x_inv = np.diag([I_base,     V_base     ])

def to_pu(Ac, Bc):
    """Scale continuous A, B into per-unit."""
    return S_x @ Ac @ S_x_inv,  S_x @ Bc * V_base

A_pu_on,  B_pu_on  = to_pu(A_on,  B_on)
A_pu_off, B_pu_off = to_pu(A_off, B_off)

print("=" * 64)
print("Per-unit continuous matrices")
print("=" * 64)
print(f"A_pu_on  =\n{A_pu_on}\n")
print(f"B_pu_on  =\n{B_pu_on}\n")
print(f"A_pu_off =\n{A_pu_off}  (identical to A_pu_on)\n")
print(f"B_pu_off =\n{B_pu_off}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Forward Euler discrete matrices in PU
#    Ad = I + A*dt,   Bd = B*dt
# ──────────────────────────────────────────────────────────────────────────────
I2 = np.eye(2)
Ad_on  = I2 + A_pu_on  * dt
Bd_on  = B_pu_on  * dt
Ad_off = I2 + A_pu_off * dt
Bd_off = B_pu_off * dt

print("=" * 64)
print("Forward Euler discrete matrices (PU, dt = T_base = 500 ns)")
print("=" * 64)
print(f"Ad_on  (= Ad_off) =\n{Ad_on}\n")
print(f"Bd_on  =\n{Bd_on}\n")
print(f"Bd_off =\n{Bd_off}\n")

eigs = np.linalg.eigvals(Ad_on)
ok   = all(abs(e) < 1.0 for e in eigs)
print(f"Ad eigenvalues: {eigs}")
print(f"  |lambda| = {np.abs(eigs)}  ->  {'stable' if ok else 'UNSTABLE'}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 5. PWM signal + piecewise Forward Euler simulation (PU domain)
# ──────────────────────────────────────────────────────────────────────────────
t_end = 10.0e-3    # 1 ms = 20 switching cycles
N     = int(t_end / dt) + 1
t_arr = np.arange(N) * dt
u_pu  = V_in / V_base

# PWM: 1 = ON for first steps_on steps within each period
sw_signal = ((np.arange(N) % steps_per) < steps_on).astype(int)

iL_pu_hist = np.zeros(N)
vC_pu_hist = np.zeros(N)
sw_hist    = np.zeros(N)

x_pu = np.array([0.0, 0.0])
for k in range(N):
    iL_pu_hist[k] = x_pu[0]
    vC_pu_hist[k] = x_pu[1]
    sw_hist[k]    = sw_signal[k]

    if sw_signal[k]:                                          # SWITCH ON
        x_pu = Ad_on  @ x_pu + Bd_on.flatten()  * u_pu
    else:                                                     # SWITCH OFF
        x_pu = Ad_off @ x_pu + Bd_off.flatten() * u_pu

# ──────────────────────────────────────────────────────────────────────────────
# 6. Convert PU -> physical
# ──────────────────────────────────────────────────────────────────────────────
iL = iL_pu_hist * I_base
vC = vC_pu_hist * V_base

ss_idx    = N - 10 * steps_per
iL_avg    = iL[ss_idx:].mean()
vC_avg    = vC[ss_idx:].mean()
iL_ripple = iL[ss_idx:].max() - iL[ss_idx:].min()
vC_ripple = vC[ss_idx:].max() - vC[ss_idx:].min()

print("=" * 64)
print("Steady-state results (avg over last 10 cycles)")
print("=" * 64)
print(f"  i_L avg  = {iL_avg:.6f} A    (theory: {iL_ss:.6f} A)")
print(f"  v_C avg  = {vC_avg:.6f} V    (theory: {vC_ss:.6f} V)")
print(f"  DeltaiL  = {iL_ripple:.4f} A    (theory: {delta_iL_th:.4f} A)")
print(f"  DeltavC  = {vC_ripple*1e3:.3f} mV   (theory: {delta_vC_th*1e3:.3f} mV)")
print(f"\n  NOTE: DeltaiL/iL_avg = {iL_ripple/iL_avg*100:.0f}%")
print(f"  Large ripple is physical for L=10uH at 20kHz.")
print(f"  Synchronous switch allows reverse i_L; average is exact.")

# ──────────────────────────────────────────────────────────────────────────────
# 7. Plot
# ──────────────────────────────────────────────────────────────────────────────
t_us  = t_arr * 1e6
BLUE   = '#1565C0'
ORANGE = '#E65100'
RED    = '#C62828'
GRAY   = '#546E7A'
GREEN  = '#2E7D32'

fig = plt.figure(figsize=(14, 15))
fig.suptitle(
    "Buck Converter — Piecewise Forward Euler  (ON / OFF matrices)\n"
    f"L = {L*1e6:.0f} uH,  C = {C*1e6:.0f} uF,  R = {R} Ohm,  "
    f"Vin = {V_in} V,  D = {D:.0%},  "
    f"f_ctrl = {f_ctrl/1e3:.0f} kHz,  f_sim = {f_sim/1e6:.0f} MHz  "
    f"(dt = {dt*1e9:.0f} ns)",
    fontsize=11, fontweight='bold'
)
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.52, wspace=0.35)

# ── Row 0: Full transient (physical) ─────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(t_us, iL, color=BLUE, lw=0.7)
ax0.axhline(iL_avg, color=GRAY,  ls='--', lw=1.0, label=f'avg = {iL_avg:.2f} A')
ax0.axhline(iL_ss,  color=GREEN, ls=':',  lw=0.9, label=f'theory = {iL_ss:.2f} A')
ax0.axhline(0,      color='k',   ls=':',  lw=0.7, label='zero')
ax0.set_title("Inductor Current — Full Transient", fontweight='semibold')
ax0.set_xlabel("Time (us)"); ax0.set_ylabel("iL (A)")
ax0.legend(fontsize=8); ax0.grid(True, alpha=0.3)

ax1 = fig.add_subplot(gs[0, 1])
ax1.plot(t_us, vC, color=ORANGE, lw=0.7)
ax1.axhline(vC_avg, color=GRAY,  ls='--', lw=1.0, label=f'avg = {vC_avg:.3f} V')
ax1.axhline(vC_ss,  color=GREEN, ls=':',  lw=0.9, label=f'theory = {vC_ss:.3f} V')
ax1.set_title("Output Voltage — Full Transient", fontweight='semibold')
ax1.set_xlabel("Time (us)"); ax1.set_ylabel("vC (V)")
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

# ── Row 1: Per-unit ───────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(t_us, iL_pu_hist, color=BLUE, lw=0.7)
ax2.axhline(iL_ss/I_base, color=GRAY, ls='--', lw=0.9,
            label=f'theory SS = {iL_ss/I_base:.3f} pu')
ax2.set_title("Inductor Current — Per-Unit", fontweight='semibold')
ax2.set_xlabel("Time (us)"); ax2.set_ylabel("iL (pu)")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(t_us, vC_pu_hist, color=ORANGE, lw=0.7)
ax3.axhline(vC_ss/V_base, color=GRAY, ls='--', lw=0.9,
            label=f'theory SS = {vC_ss/V_base:.3f} pu')
ax3.set_title("Output Voltage — Per-Unit", fontweight='semibold')
ax3.set_xlabel("Time (us)"); ax3.set_ylabel("vC (pu)")
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

# ── Row 2: Steady-state zoom (last 5 cycles) ─────────────────────────────────
zoom_s = N - 5 * steps_per
t_z    = t_us[zoom_s:]
iL_z   = iL[zoom_s:]
vC_z   = vC[zoom_s:]
sw_z   = sw_hist[zoom_s:]

ax4 = fig.add_subplot(gs[2, 0])
ax4r = ax4.twinx()
ax4r.fill_between(t_z, sw_z, step='post', alpha=0.12, color=RED)
ax4r.set_ylim(-0.1, 3.0); ax4r.set_yticks([0, 1])
ax4r.set_yticklabels(['OFF', 'ON'], color=RED, fontsize=7)
ax4.plot(t_z, iL_z, color=BLUE, lw=1.3)
ax4.axhline(iL_avg, color=GRAY, ls='--', lw=0.9, label=f'avg = {iL_avg:.2f} A')
ax4.axhline(0,      color='k',  ls=':',  lw=0.7, label='zero')
ax4.set_title(f"iL Steady-State Zoom  (Delta = {iL_ripple:.2f} A p-p)", fontweight='semibold')
ax4.set_xlabel("Time (us)"); ax4.set_ylabel("iL (A)")
ax4.legend(loc='lower right', fontsize=8); ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[2, 1])
ax5r = ax5.twinx()
ax5r.fill_between(t_z, sw_z, step='post', alpha=0.12, color=RED)
ax5r.set_ylim(-0.1, 3.0); ax5r.set_yticks([0, 1])
ax5r.set_yticklabels(['OFF', 'ON'], color=RED, fontsize=7)
ax5.plot(t_z, vC_z, color=ORANGE, lw=1.3)
ax5.axhline(vC_avg, color=GRAY, ls='--', lw=0.9, label=f'avg = {vC_avg:.3f} V')
ax5.set_title(f"vC Steady-State Zoom  (Delta = {vC_ripple*1e3:.1f} mV p-p)", fontweight='semibold')
ax5.set_xlabel("Time (us)"); ax5.set_ylabel("vC (V)")
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)

# ── Row 3: Single cycle annotated ────────────────────────────────────────────
cyc_s  = N - 2 * steps_per
cyc_e  = cyc_s + steps_per
t_c    = t_us[cyc_s:cyc_e]
iL_c   = iL[cyc_s:cyc_e]
vC_c   = vC[cyc_s:cyc_e]
sw_c   = sw_hist[cyc_s:cyc_e]

on_idx  = np.where(sw_c == 1)[0]
off_idx = np.where(sw_c == 0)[0]
t_mid_on  = t_c[on_idx[len(on_idx)//2]]
t_mid_off = t_c[off_idx[len(off_idx)//2]]

ax6 = fig.add_subplot(gs[3, 0])
ax6r = ax6.twinx()
ax6r.fill_between(t_c, sw_c, step='post', alpha=0.13, color=RED)
ax6r.set_ylim(-0.1, 3.0); ax6r.set_yticks([0, 1])
ax6r.set_yticklabels(['OFF', 'ON'], color=RED, fontsize=7)
ax6.plot(t_c, iL_c, color=BLUE, lw=1.6)
ax6.axhline(0, color='k', ls=':', lw=0.7)
y_hi = iL_c.max() * 0.75
y_lo = iL_c.min() * 0.6
ax6.annotate('Ad_on\nBd_on',   xy=(t_mid_on,  y_hi), ha='center', fontsize=8,
             color=RED,  fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.25', fc='#FFEBEE', alpha=0.85))
ax6.annotate('Ad_off\nBd_off', xy=(t_mid_off, y_lo), ha='center', fontsize=8,
             color=BLUE, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.25', fc='#E3F2FD', alpha=0.85))
ax6.set_title("Single Cycle — iL  (matrix selection annotated)", fontweight='semibold')
ax6.set_xlabel("Time (us)"); ax6.set_ylabel("iL (A)")
ax6.grid(True, alpha=0.3)

ax7 = fig.add_subplot(gs[3, 1])
ax7r = ax7.twinx()
ax7r.fill_between(t_c, sw_c, step='post', alpha=0.13, color=RED)
ax7r.set_ylim(-0.1, 3.0); ax7r.set_yticks([0, 1])
ax7r.set_yticklabels(['OFF', 'ON'], color=RED, fontsize=7)
ax7.plot(t_c, vC_c, color=ORANGE, lw=1.6)
vC_hi = vC_c.max() - (vC_c.max()-vC_c.mean())*0.35
vC_lo = vC_c.min() + (vC_c.mean()-vC_c.min())*0.35
ax7.annotate('Ad_on\nBd_on',   xy=(t_mid_on,  vC_hi), ha='center', fontsize=8,
             color=RED,  fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.25', fc='#FFEBEE', alpha=0.85))
ax7.annotate('Ad_off\nBd_off', xy=(t_mid_off, vC_lo), ha='center', fontsize=8,
             color=BLUE, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.25', fc='#E3F2FD', alpha=0.85))
ax7.set_title("Single Cycle — vC  (matrix selection annotated)", fontweight='semibold')
ax7.set_xlabel("Time (us)"); ax7.set_ylabel("vC (V)")
ax7.grid(True, alpha=0.3)

plt.savefig("D:/ta2/buck_ss_simulation.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved.")
