import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Base Values (Per-Unit System)
# =============================================================================
V_base = 5          # V
I_base = 5          # A
f_base = 10e6        # Hz  — simulation timestep rate
T_base = 1 / f_base # s   — forward Euler timestep
Z_base = V_base / I_base
k      = I_base / V_base  # admittance base (I_base / V_base)

# =============================================================================
# Component Values
# =============================================================================
Vin      = 5        # V
R1       = 100        # Ω
L1       = 1e-4     # H
C1       = 2.2e-4   # F

# Ćuk-specific
R2       = 100        # Ω
L2       = 1e-4     # H
C2       = 2.2e-4   # F

# Fixed-point fractional bit width
N_FRAC   = 24

# =============================================================================
# Derived Scaling Coefficients
# alpha_X = T_base / X  — appears in Forward Euler state update
# =============================================================================
u      = Vin / V_base   # input voltage, per unit
aL1    = T_base / L1
aC1    = T_base / C1
aL2    = T_base / L2
aC2    = T_base / C2

# =============================================================================
# Utility Functions
# =============================================================================
def quantize(mat, n_bits):
    """
    Round each element of mat to the nearest fixed-point value
    with n_bits fractional bits. Works on nested lists or numpy arrays.
    Returns a numpy float64 array (copy — input is never mutated).
    """
    m     = np.array(mat, dtype=float)
    scale = 2 ** n_bits
    return np.round(m * scale) / scale


def stability(mat):
    """
    Forward Euler stability check.

    The state update is:  x[k+1] = (I + A) * x[k]
    so the iteration matrix is (I + A).
    Stability requires all eigenvalues of (I + A) to satisfy |λ| < 1.

    Returns
    -------
    label  : str   — 'STABLE' | 'MARGINALLY STABLE' | 'UNSTABLE'
    eigvals: ndarray of complex eigenvalues of (I + A)
    margins: ndarray of |λ| for each eigenvalue
    """
    A    = np.array(mat, dtype=float)
    Phi  = np.eye(A.shape[0]) + A          # iteration matrix
    ev   = np.linalg.eigvals(Phi)
    mag  = np.abs(ev)

    if np.all(mag < 1.0):
        label = "STABLE"
    elif np.any(mag > 1.0):
        label = "UNSTABLE"
    else:
        label = "MARGINALLY STABLE"

    return label, ev, mag


def print_matrix(name, mat):
    """Pretty-print a matrix with a header label."""
    print(f"\n  {name}:")
    for row in np.array(mat):
        print("   ", "  ".join(f"{v:12.6f}" for v in row))


def report(converter_name, substates):
    """
    Print a stability report for one converter.

    substates : list of (label, A_matrix) tuples
                e.g. [("ON", Aon), ("OFF", Aoff)]
    """
    print(f"\n{'='*60}")
    print(f"  {converter_name}")
    print(f"{'='*60}")
    for state_label, A in substates:
        A_q              = quantize(A, N_FRAC)
        label, ev, mag   = stability(A_q)
        print(f"\n  [{state_label} STATE]  →  {label}")
        print(f"  Iteration matrix eigenvalue magnitudes: {np.round(mag, 8)}")
        print(f"  Stability margin (1 - max|λ|): {1.0 - np.max(mag):.6e}")
        print_matrix(f"A_{state_label.lower()} (quantized)", A_q)

# =============================================================================
# Buck Converter
# State: x = [iL, vC]  (per unit)
# ON  : switch closed, diode off  — same LC loop as OFF (ideal diode conducts)
# OFF : switch open,  diode on    — identical network for ideal buck
# =============================================================================
Aon_buck = [
    [      0,        -aL1 ],
    [ k*aC1, -aC1 / R1   ],
]
Bon_buck = [
    [ aL1 ],
    [   0 ],
]

Aoff_buck = Aon_buck          # ideal buck: diode conducts same LC loop
Boff_buck = [[0], [0]]

# =============================================================================
# Boost Converter
# State: x = [iL, vC]  (per unit)
# ON  : switch closed — inductor charges, capacitor isolated from inductor
# OFF : switch open   — inductor + capacitor + load form series loop
# =============================================================================
Aon_boost = [
    [      0,           0 ],
    [      0, -aC1 / R1   ],
]
Bon_boost = [
    [ aL1 ],
    [   0 ],
]

Aoff_boost = [
    [      0,        -aL1 ],
    [ k*aC1, -aC1 / R1   ],
]
Boff_boost = Bon_boost

# =============================================================================
# Buck-Boost Converter
# State: x = [iL, vC]  (per unit)
# ON  : switch closed — same topology as boost ON (capacitor isolated)
# OFF : switch open   — inductor discharges into capacitor+load (inverted)
# =============================================================================
Aon_bb = [
    [      0,           0 ],
    [      0, -aC1 / R1   ],
]
Bon_bb = [
    [ aL1 ],
    [   0 ],
]

Aoff_bb = [
    [      0,        -aL1 ],
    [ k*aC1, -aC1 / R1   ],
]
Boff_bb = [[0], [0]]

# =============================================================================
# Ćuk Converter
# State: x = [iL1, iL2, vC1, vC2]  (per unit)
# ON  : switch closed
# OFF : switch open
# =============================================================================
Aon_cuk = [
    [       0,        0,   -aL1,              0 ],
    [       0,        0,      0,           aL2  ],
    [ k*aC1,          0,      0,              0 ],
    [       0,  k*aC2,         0, -aC2 / R2     ],
]
Bon_cuk = [
    [ aL1 ],
    [   0 ],
    [   0 ],
    [   0 ],
]

Aoff_cuk = [
    [       0,        0,      0,              0 ],
    [       0,        0,   aL2,           -aL2  ],
    [       0, -k*aC1,         0,              0 ],
    [       0,  k*aC2,         0, -aC2 / R2     ],
]
Boff_cuk = Bon_cuk

# =============================================================================
# Run Stability Reports
# =============================================================================
if __name__ == "__main__":
    print(f"\nForward Euler timestep : {T_base:.2e} s")
    print(f"Fixed-point fractional : {N_FRAC} bits")
    print(f"LSB                    : {2**-N_FRAC:.2e}")

    report("BUCK CONVERTER",      [("ON",  Aon_buck),  ("OFF", Aoff_buck)])
    report("BOOST CONVERTER",     [("ON",  Aon_boost), ("OFF", Aoff_boost)])
    report("BUCK-BOOST CONVERTER",[("ON",  Aon_bb),    ("OFF", Aoff_bb)])
    report("CUK CONVERTER",       [("ON",  Aon_cuk),   ("OFF", Aoff_cuk)])

    print(f"\n{'='*60}\n")


val_on = np.linalg.eigvals(Aon_cuk)
val_off = np.linalg.eigvals(Aoff_cuk)

fig, ax = plt.subplots()
circle_back = plt.Circle((1, 0), 1, color='black', fill=False)
circle_fow = plt.Circle((-1,0), 1, color='black', fill = False)
# Add circle to the plot
ax.add_patch(circle_back)
ax.add_patch(circle_fow)

# Set equal scaling and limits so it looks like a circle
ax.set_aspect('equal')
for w in val_on:
    plt.scatter(w.real, w.imag, color='blue', marker='x')
for w in val_off:
    plt.scatter(w.real, w.imag, color='green', marker='x')
# Plot complex eigenvalues
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.title("Eigenvalues in Complex Plane")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.show()
