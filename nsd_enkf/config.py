"""
config.py
=========
Single source of truth for all configuration, constants, and model parameters.

Change RUN_NAME to version a new experiment.
"""

from pathlib import Path

# ─── Run identity ─────────────────────────────────────────────────────────────
RUN_NAME = "tuned_v1"

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results" / RUN_NAME

# ─── Dataset configuration ───────────────────────────────────────────────────
DATASETS_ALL = {
    f"P{i}": {
        "file": DATA_DIR / f"P{i}.xlsx",
        "sheets": {
            "met": "Metabolites",
            "nsd": "NSD",
        },
    }
    for i in range(1, 5)
}

# ─── State vector definition ────────────────────────────────────────────────
STATE_NAMES = [
    "Xv", "mAb", "Gal", "Urd", "Glc", "Amm", "Gln", "Lac",
    "Asn", "Glu",
    "UDPGal", "UDPGalNAc", "UDPGlc", "UDPGlcNAc",
    "GDPMan", "GDPFuc", "CMPNeu5Ac",
]
STATE_NUM = len(STATE_NAMES)

AXIS_NAMES = [
    r"Viable Cell Density (cell L$^{-1}$)",
    r"mAb Titre (mg L$^{-1}$)",
    r"Galactose Concentration (mM)",
    r"Uridine Concentration (mM)",
    r"Glucose Concentration (mM)",
    r"Ammonia Concentration (mM)",
    r"Glutamine Concentration (mM)",
    r"Lactate Concentration (mM)",
    r"Asparagine Concentration (mM)",
    r"Glutamate Concentration (mM)",
    r"UDP-Gal Concentration (mM)",
    r"UDP-GalNAc Concentration (mM)",
    r"UDP-Glc Concentration (mM)",
    r"UDP-GlcNAc Concentration (mM)",
    r"GDP-Man Concentration (mM)",
    r"GDP-Fuc Concentration (mM)",
    r"CMP-Neu5Ac Concentration (mM)",
]

RMSE_NAMES = [
    r"RMSE-Viable Cell Density (cell L$^{-1}$)",
    r"RMSE-mAb (mg L$^{-1}$)",
    r"RMSE-Gal (mM)",
    r"RMSE-Urd (mM)",
    r"RMSE-Glucose (mM)",
    r"RMSE-Ammonia (mM)",
    r"RMSE-Gln (mM)",
    r"RMSE-Lac (mM)",
    r"RMSE-Asn (mM)",
    r"RMSE-Glu (mM)",
    r"RMSE-UDPGal (mM)",
    r"RMSE-UDPGalNAc (mM)",
    r"RMSE-UDPGlc (mM)",
    r"RMSE-UDPGlcNAc (mM)",
    r"RMSE-GDPMan (mM)",
    r"RMSE-GDPFuc (mM)",
    r"RMSE-CMPNeu5Ac (mM)",
]

# Measured states used in EnKF update (extracellular only, no Asn/NSD)
MEASURED_STATES = ["Xv", "mAb", "Gal", "Urd", "Glc", "Amm", "Gln", "Lac"]
MEAS_NUM = len(MEASURED_STATES)

# ─── Simulation time grid ───────────────────────────────────────────────────
DT = 0.01        # hours
T_END = 288.0    # hours (12 days)

# ─── Feed / sampling schedule ───────────────────────────────────────────────
FIN_PULSES = [48.02, 96.02, 144.02, 192.02, 240.02]  # hours

FOUT_PULSES = {
    48.01: 0.63,
    72.01: 0.37,
    96.01: 0.30,
    96.03: 0.36,
    120.01: 0.24,
    144.01: 0.26,
    144.03: 0.10,
    168.01: 0.54,
    192.01: 0.23,
    192.03: 0.10,
    216.01: 0.53,
    240.01: 0.23,
    240.03: 0.10,
    264.00: 0.53,
}

BOLUS_DOSES = {
    "P1": {
        96.02:  (79.35, 15.87),
        144.02: (15.38, 3.08),
        192.02: (10.99, 2.20),
        240.02: (248.29, 49.66),
    },
    "P2": {
        96.02:  (4.27, 0.85),
        144.02: (168.34, 33.67),
        192.02: (37.72, 7.54),
        240.02: (11.35, 2.27),
    },
    "P3": {
        96.02:  (5.19, 1.04),
        144.02: (3.11, 0.62),
        192.02: (235.29, 47.06),
        240.02: (249.94, 49.99),
    },
    "P4": {
        96.02:  (21.91, 4.38),
        144.02: (6.41, 1.28),
        192.02: (233.46, 46.69),
        240.02: (3.97, 0.79),
    },
}

# Fixed measurement times (shared across datasets)
T_MEAS_FIXED = [
    0., 24.01, 48.01, 72.01, 96., 96.03, 120.01, 144.01, 144.03,
    168.01, 192.01, 192.03, 216.01, 240.01, 240.03, 264.01, 288.0,
]

# ─── Initial volumes (L) ────────────────────────────────────────────────────
INITIAL_VOLUMES = {
    "P1": 0.1,
    "P2": 0.1,
    "P3": 0.1,
    "P4": 0.1,
}

# ─── Model parameters ───────────────────────────────────────────────────────
# Cell and product
V_CELL = 1.123e-12     # L/v.cell
MW = 165174             # g_prod/mol_prod
ACOMP = 9.915651e-9

# Controller parameters
SP = 71
K_C = 2.572248e-6
T_I = 176.0905
T_D = 290.347

# Growth and death kinetics
MU_MAX = 0.065
MU_D_MAX = 0.015
KLYSIS = 0.5

# Monod/inhibition constants
KGLC = 14.0378
KLAC = 0.00001
KASN = 2.62371
KGLU = 0.000001
KGLN = 0.00000454277
KILAC = 1000
KIAMM = 3.16935
KIURD = 41.0875
KIGAL = 1000
KD_AMM = 14.2830
KD_URD = 27.8752
KGAL = 18.2317
KURD = 7.00810

# Maintenance coefficients
M_LAC = 1.87253e-10    # mmol/cell/h
M_GLC = 3.43293e-11
KC_GAL = 5.27033
LAC_MAX_1 = 21.1983
LAC_MAX_2 = 16
FACTOR = 0.347987069
NG = 1

# Feed concentrations (mM)
LAC_FEED = 0.
GLC_FEED = 144.37
GLU_FEED = 12.19
ASP_FEED = 51.95
ARG_FEED = 9.16
ASN_FEED = 26.99
LYS_FEED = 16.64
PRO_FEED = 10.18
GLN_FEED = 0.
AMM_FEED = 0.06

# Yield coefficients (cell/mmol)
YX_GLC = 1.0115e9
YX_GLN = 4.64127e9
YX_GLU = 1.45647e10
YX_LAC = 5.45539e7
YX_AMM = 2.36299e9
YX_GAL = 1.38498e8
YX_URD = 1.61202e9
YX_ASN = 7.6824e8
YX_ASP = 3.59e9
YX_ARG = 2.64e10
YX_LYS = 1.75e10
YX_PRO = 3.26e11

# Cross-yield coefficients (mmol/mmol)
YGLN_GLU = 0.
YGLN_ASN = 0.
YGLN_AMM = 0.104524
YLAC_GLC = 1.56
YASP_ASN = 0.126
YARG_GLU = 0.007
YLYS_GLU = 0.116
YPRO_GLU = 1.
YAMM_URD = 2.
YASN_ASP = 0.1

# mAb production
YMAB_XV = 4.12718e-10
YMAB_MU = 3.38956e-9

# ─── NSD pathway parameters ─────────────────────────────────────────────────
# Intracellular glutamine fraction
F2 = 0.0222435

# Maximum velocities (mmol/L/h) — main NSD pathway
VMAX1 = 0.921507
VMAX2 = 0.0169968
VMAX2B = 59.4891
VMAX3 = 0.0550887
VMAX4 = 0.0265253
VMAX5 = 0.0001
VMAX6 = 5.1304
VMAX7 = 4.59677

# Dissociation constants (mM) — main NSD pathway
KDF_E1_GLN = 0.418760
KDF_E2B = 0.0248298
KDF_E4 = 2.31278
KDF_E5 = 0.0269656
KDF_E6 = 0.0163559
KDF_E7 = 0.994547
KDF_GLC_UDPGLC = 78.1241
KDF_GLC_GDPMAN = 50.

# Inhibition constants (mM) — main NSD pathway
KIE2A = 1.04504e-6
KIE2B = 92.1059
KIE2C = 0.0132697
KIE2D = 2.66102e-6
KIE5 = 1000.
KIE6A = 1.10182e-7
KIE6B = 4.57309
KIE6C = 4.83505e-6
KI_E7 = 0.0164192

# Gal/Urd feeding rates
VMAX1U = 0.147995
VMAX2U = 0.0451806
VMAX4U = 0.0127551
VMAX6U = 5.34270
VMAX6G = 40.8965

K1U = 6.08196
K2U = 13.6332
K4U = 6.24826
K6U = 0.438499
K6G = 0.600019

KI6_URD = 0.000911002
KI6_GLC = 0.292793
KI6_GAL = 99.6298
KI6_UGAL = 0.01

# Sink rates
VMAX6_SINK = 7.30429
VMAX7_SINK = 10.9370
VMAX1_SINK = 25.4859

K6_SINK = 0.128756
K7_SINK = 8.87794
K1_SINK = 0.0406881

KI1_SINK = 0.000120640

# NSD transport Monod constants
KTP_UDPGLC = 0.989957
KTP_UDPGAL = 7.1464
KTP_UDPGLCNAC = 5.04905
KTP_UDPGALNAC = 11.0558
KTP_GDPMAN = 0.127219
KTP_GDPFUC = 0.1
KTP_CMPNEU5AC = 503.213

# NSD consumption per cell for HCP/lipids (mmolNSD/cell)
NHCP_LIPIDS_UDPGLC = 1.560e-12
NHCP_LIPIDS_UDPGLCNAC = 1.248e-12
NHCP_LIPIDS_UDPGAL = 2.288e-12
NHCP_LIPIDS_UDPGALNAC = 1.252e-12
NHCP_LIPIDS_GDPMAN = 3.538e-12
NHCP_LIPIDS_GDPFUC = 0.140e-12
NHCP_LIPIDS_CMPNEU5AC = 1.846e-12

# NSD consumption per mg mAb (mmolNSD/mgmAb)
NMAB_UDPGLC = 40.39e-6
NMAB_UDPGLCNAC = 26.67e-6
NMAB_UDPGLCNAC_B = 49.14e-6 - 26.67e-6
NMAB_UDPGAL = 7.119e-6
NMAB_UDPGALNAC = 0.
NMAB_GDPMAN = 121.2e-6
NMAB_GDPFUC = 12.23e-6
NMAB_CMPNEU5AC = 0.155e-6

# ─── EnKF noise parameters ──────────────────────────────────────────────────
# Two noise modes are supported:
#   - Multiplicative (state-proportional): noise_i ~ N(0, (cv_i * x_i)^2)
#     Used for measured extracellular states. Naturally gives zero noise when
#     concentration is zero, and scales with state magnitude.
#   - Additive (fixed variance): noise_i ~ N(0, Q_ii)
#     Used for unmeasured states where we lack diagnostics to calibrate CV.
#
# PROCESS_NOISE_CV: coefficient of variation for multiplicative noise states.
# These are dimensionless fractions (e.g., 0.03 = 3% CV per timestep).
# Tuned on P4 via innovation-based diagnostics.
# CV values are per-step (dt=0.01h). They compound as cv*sqrt(N_steps) between
# measurement updates (~2400 steps = 24h apart), so 0.002/step ≈ 10% between updates.
PROCESS_NOISE_CV = {
    # Measured extracellular (tuned via innovation diagnostics on P4)
    'Xv':  0.008,   # ~40% between updates
    'mAb': 0.008,   # ~40% between updates
    'Gal': 0.006,   # ~30% between updates
    'Urd': 0.008,   # ~40% between updates
    'Glc': 0.010,   # ~50% between updates (structural model bias)
    'Amm': 0.008,   # ~40% between updates
    'Gln': 0.005,   # ~25% between updates
    'Lac': 0.009,   # ~45% between updates (structural model bias)
}

# PROCESS_NOISE_VAR: additive noise variance for states NOT in PROCESS_NOISE_CV.
# Unmeasured states use additive noise (multiplicative is unstable without
# measurement correction to prevent ensemble divergence).
PROCESS_NOISE_VAR = {
    'Xv': 0, 'mAb': 0, 'Gal': 0, 'Urd': 0,
    'Glc': 0, 'Amm': 0, 'Gln': 0, 'Lac': 0,
    'Asn': 1e-5, 'Glu': 4.8e-5,
    'UDPGal': 2e-4, 'UDPGalNAc': 1e-5, 'UDPGlc': 6e-5,
    'UDPGlcNAc': 1e-5, 'GDPMan': 2e-7, 'GDPFuc': 2e-7,
    'CMPNeu5Ac': 2e-4,
}

# Measurement noise variance R: set from experimental error bars (biological
# triplicate variance, mean across P1-P4 datasets). These represent the
# observed variability across biological replicates and provide a conservative
# upper bound on measurement uncertainty.
MEASUREMENT_NOISE_VAR = {
    'Xv': 6.928e+16, 'mAb': 573.3, 'Gal': 1.739, 'Urd': 5.103e-3,
    'Glc': 1.256, 'Amm': 4.500e-2, 'Gln': 7.350e-3, 'Lac': 0.2971,
}

# Initial ensemble covariance P0: for measured states, set from measurement
# error bar variance; for unmeasured states, set from process noise variance.
# Separate from per-step Q to ensure the ensemble starts with meaningful spread.
INITIAL_COV_OVERRIDE = {
    'Xv': 6.928e+16, 'mAb': 573.3, 'Gal': 1.739, 'Urd': 5.103e-3,
    'Glc': 1.256, 'Amm': 4.500e-2, 'Gln': 7.350e-3, 'Lac': 0.2971,
}

ENSEMBLE_SIZE = 100
KQ = 1.0   # Q = KQ * diag(PROCESS_NOISE_VAR); only affects additive states
N_RUNS = 10

# ─── Dataset display customisation ──────────────────────────────────────────
DATASET_COLOURS = {
    "P1": "tab:orange",
    "P2": "tab:green",
    "P3": "tab:blue",
    "P4": "tab:red",
}

DATASET_MARKERS = {
    "P1": "o",
    "P2": "s",
    "P3": "^",
    "P4": "D",
}
