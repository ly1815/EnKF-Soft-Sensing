"""
model.py
========
CHO bioprocess model with NSD pathway: volume integration, kinetic rates,
ODE-based model step (LSODA), and open-loop simulation.
"""

import numpy as np
import scipy.integrate as scp
from tqdm import tqdm

from nsd_enkf.config import (
    # Growth and death
    MU_MAX, MU_D_MAX, KGLC, KASN, KILAC, KIAMM, KIURD, KD_AMM, KD_URD,
    KGAL, KURD, KC_GAL, FACTOR,
    # Yield and maintenance
    YX_GLC, YX_GLN, YX_GLU, YX_LAC, YX_AMM, YX_GAL, YX_URD, YX_ASN, YX_ASP,
    M_LAC, M_GLC, LAC_MAX_1, LAC_MAX_2,
    YGLN_GLU, YGLN_ASN, YGLN_AMM, YLAC_GLC, YASP_ASN, YAMM_URD, YASN_ASP,
    YMAB_XV, YMAB_MU,
    # Feed concentrations
    GLC_FEED, GLU_FEED, ASN_FEED, GLN_FEED, AMM_FEED, LAC_FEED,
    # NSD
    V_CELL, F2,
    VMAX1, VMAX2, VMAX2B, VMAX3, VMAX4, VMAX5, VMAX6, VMAX7,
    KDF_E1_GLN, KDF_E2B, KDF_E4, KDF_E5, KDF_E6, KDF_E7, KDF_GLC_UDPGLC, KDF_GLC_GDPMAN,
    KIE2A, KIE2B, KIE2C, KIE2D, KIE5, KIE6A, KIE6B, KIE6C, KI_E7,
    VMAX1U, VMAX2U, VMAX4U, VMAX6U, VMAX6G,
    K1U, K2U, K4U, K6U, K6G,
    KI6_URD, KI6_GLC, KI6_GAL, KI6_UGAL,
    VMAX6_SINK, VMAX7_SINK, VMAX1_SINK,
    K6_SINK, K7_SINK, K1_SINK, KI1_SINK,
    KTP_UDPGLC, KTP_UDPGAL, KTP_UDPGLCNAC, KTP_UDPGALNAC,
    KTP_GDPMAN, KTP_GDPFUC, KTP_CMPNEU5AC,
    NHCP_LIPIDS_UDPGLC, NHCP_LIPIDS_UDPGLCNAC, NHCP_LIPIDS_UDPGAL,
    NHCP_LIPIDS_UDPGALNAC, NHCP_LIPIDS_GDPMAN, NHCP_LIPIDS_GDPFUC, NHCP_LIPIDS_CMPNEU5AC,
    NMAB_UDPGLC, NMAB_UDPGLCNAC, NMAB_UDPGLCNAC_B, NMAB_UDPGAL,
    NMAB_UDPGALNAC, NMAB_GDPMAN, NMAB_GDPFUC, NMAB_CMPNEU5AC,
)


# ─── Helper functions ────────────────────────────────────────────────────────

def _to_1d(x):
    return np.asarray(x, dtype=float).reshape(-1)


def _to_scalar(x):
    return np.asarray(x).reshape(-1)[0].item()


# ─── Volume integration ──────────────────────────────────────────────────────

def volume_integration(init_volume, Fin, Fout, step_len):
    """
    Integrate bioreactor volume over time using dV/dt = Fin - Fout.

    Returns
    -------
    np.ndarray  Volume at each time point (length = len(Fin) + 1).
    """
    def volume_model(t, state, fin, fout):
        return np.array([fin - fout], dtype="float64")

    current_volume = _to_scalar(init_volume)
    volumes = [current_volume]

    for step_i in tqdm(range(len(Fin)), desc="Integrating Volume", leave=False):
        fin = _to_scalar(Fin[step_i])
        fout = _to_scalar(Fout[step_i])

        ode = scp.ode(volume_model).set_integrator("lsoda", nsteps=3000)
        ode.set_initial_value(current_volume, 0.0).set_f_params(fin, fout)

        new_volume = ode.integrate(ode.t + _to_scalar(step_len[step_i]))
        current_volume = _to_scalar(new_volume)
        volumes.append(current_volume)

    return np.array(volumes)


def compute_volume_results(datasets, initial_volumes, build_schedule_fn, step_len):
    """
    Compute volume profiles for all datasets.

    Returns
    -------
    dict  {dataset_name: np.ndarray}
    """
    volume_results = {}
    for name in datasets:
        Fin, Fout, _, _ = build_schedule_fn(name)
        init_vol = initial_volumes.get(name, 0.1)
        volume_results[name] = volume_integration(init_vol, Fin, Fout, step_len)
        print(f"Volume integration complete: {name}")
    return volume_results


# ─── Kinetic rate expressions ────────────────────────────────────────────────

def model_params(state, ng=1):
    """
    Compute kinetic rates and NSD fluxes from the current state.

    Returns a tuple of all rates used by the ODE system.
    """
    (Xv, mAb, Gal, Urd, Glc, Amm, Gln, Lac, Asn, Glu,
     UDPGal, UDPGalNAc, UDPGlc, UDPGlcNAc, GDPMan, GDPFuc, CMPNeu5Ac) = state

    # Growth and death rate
    flim = Glc / (KGLC + Glc) * Asn / (KASN + Asn)
    finh = (KIAMM / (KIAMM + Amm)) * (KILAC / (KILAC + Lac)) * (KIURD / (KIURD + Urd))
    mu = MU_MAX * flim * finh
    mu_d = MU_D_MAX * (Amm / (KD_AMM + Amm) + Urd / (KD_URD + Urd))

    # Metabolite rates
    Qgal = (-mu / YX_GAL) * (Gal / (Gal + KGAL))
    Qurd = -Urd / (KURD + Urd) * mu / YX_URD

    Qglc = (-mu / YX_GLC - M_GLC) * (KC_GAL / (KC_GAL + Gal)) ** ng
    ng_local = 1 - FACTOR * (Qgal / Qglc) if Qglc != 0 else 1

    Qamm = mu / YX_AMM - YAMM_URD * Qurd
    Qlac = ((mu / YX_LAC - YLAC_GLC * Qglc) * (LAC_MAX_1 - Lac) / LAC_MAX_1
            + M_LAC * (LAC_MAX_2 - Lac) / LAC_MAX_2)
    Qglu = -mu / YX_GLU
    Qasn = ((mu * (YX_ASP - YX_ASN * YASP_ASN)) / (YX_ASN * YX_ASP)
            * (YASP_ASN * YASN_ASP - 1))
    Qgln = mu / YX_GLN - Qglu * YGLN_GLU - Qasn * YGLN_ASN + YGLN_AMM * Qamm
    Gln_int = F2 * Gln
    QmAb = YMAB_MU * mu + YMAB_XV

    # NSD outflux (transport + glycosylation demand)
    Fout_UDPGal = (UDPGal / (KTP_UDPGAL + UDPGal)
                   * (NHCP_LIPIDS_UDPGAL * mu / V_CELL + NMAB_UDPGAL * QmAb / V_CELL))
    Fout_UDPGalNAc = (UDPGalNAc / (KTP_UDPGALNAC + UDPGalNAc)
                      * (NHCP_LIPIDS_UDPGALNAC * mu / V_CELL + NMAB_UDPGALNAC * QmAb / V_CELL))
    Fout_UDPGlc = (UDPGlc / (KTP_UDPGLC + UDPGlc)
                   * (NHCP_LIPIDS_UDPGLC * mu / V_CELL + NMAB_UDPGLC * QmAb / V_CELL))
    Fout_UDPGlcNAc = (UDPGlcNAc / (KTP_UDPGLCNAC + UDPGlcNAc)
                      * (NHCP_LIPIDS_UDPGLCNAC * mu / V_CELL
                         + NMAB_UDPGLCNAC * QmAb / V_CELL
                         + NMAB_UDPGLCNAC_B * QmAb / V_CELL))
    Fout_GDPMan = (GDPMan / (KTP_GDPMAN + GDPMan)
                   * (NHCP_LIPIDS_GDPMAN * mu / V_CELL + NMAB_GDPMAN * QmAb / V_CELL))
    Fout_GDPFuc = (GDPFuc / (KTP_GDPFUC + GDPFuc)
                   * (NHCP_LIPIDS_GDPFUC * mu / V_CELL + NMAB_GDPFUC * QmAb / V_CELL))
    Fout_CMPNeu5Ac = (CMPNeu5Ac / (KTP_CMPNEU5AC + CMPNeu5Ac)
                      * (NHCP_LIPIDS_CMPNEU5AC * mu / V_CELL + NMAB_CMPNEU5AC * QmAb / V_CELL))

    # NSD enzymatic rates — main pathway
    r1_f = VMAX1 * Gln_int / (KDF_E1_GLN + Gln_int)
    r2_f = VMAX2 * Glc / (KDF_GLC_UDPGLC + Glc)
    r2_bf = VMAX2B * UDPGal / (KDF_E2B * (1 + UDPGlcNAc / KIE2A + UDPGalNAc / KIE2B
                                            + UDPGlc / KIE2C + UDPGal / KIE2D) + UDPGal)
    r3_f = VMAX3 * Glc / (KDF_GLC_GDPMAN + Glc)
    r4_f = VMAX4 * UDPGlcNAc / (KDF_E4 + UDPGlcNAc)
    r5_f = VMAX5 * UDPGlcNAc / (UDPGlcNAc + KDF_E5 * (1 + CMPNeu5Ac / KIE5))

    r6_f = (VMAX6 * UDPGlc) / (KDF_E6 * (1 + UDPGlcNAc / KIE6A + UDPGalNAc / KIE6B
                                           + UDPGal / KIE6C) + UDPGlc)
    r6_gal = VMAX6G * Gal / (K6G * (1 + UDPGal / KI6_UGAL + Gal / KI6_GAL
                                      + Urd / KI6_URD) + Gal)
    r7_f = VMAX7 * GDPMan / ((KDF_E7 + GDPMan) * (1 + GDPFuc / KI_E7))
    r7_sink = VMAX7_SINK * (GDPFuc / (GDPFuc + K7_SINK))
    r1_sink = VMAX1_SINK * UDPGlcNAc / ((UDPGlcNAc + K1_SINK) * (1 + CMPNeu5Ac / KI1_SINK))
    r6_sink = VMAX6_SINK * UDPGal / (UDPGal + K6_SINK * (1 + UDPGlc / KI6_GLC)) * (Gal / (Gal + 0.00001))

    # Uridine rates
    r1_urd = VMAX1U * Urd / (K1U + Urd)
    r2_urd = VMAX2U * Urd / (K2U + Urd)
    r4_urd = VMAX4U * Urd / (K4U + Urd)
    r6_urd = VMAX6U * Urd / (K6U + Urd)

    return (mu, mu_d, Qgal, Qurd, Qglc, Qamm, Qgln, Qlac, Qglu, Qasn, Gln_int, QmAb,
            Fout_UDPGal, Fout_UDPGalNAc, Fout_UDPGlc, Fout_UDPGlcNAc,
            Fout_GDPMan, Fout_GDPFuc, Fout_CMPNeu5Ac,
            r1_f, r2_f, r2_bf, r3_f, r4_f, r5_f, r6_f, r6_gal, r7_f,
            r7_sink, r1_sink, r6_sink,
            r1_urd, r2_urd, r4_urd, r6_urd)


# ─── Model step (single integration step) ───────────────────────────────────

def _dynamic_model(t, state, Fin, Fout_ctrl, V, Gal_feed, Urd_feed):
    """RHS of the full mechanistic ODE (module-level so a single integrator can be
    reused across calls; controls are passed via ode.set_f_params)."""
    (Xv, mAb, Gal, Urd, Glc, Amm, Gln, Lac, Asn, Glu,
     UDPGal, UDPGalNAc, UDPGlc, UDPGlcNAc, GDPMan, GDPFuc, CMPNeu5Ac) = state

    (mu, mu_d, Qgal, Qurd, Qglc, Qamm, Qgln, Qlac, Qglu, Qasn, Gln_int, QmAb,
     Fout_UDPGal, Fout_UDPGalNAc, Fout_UDPGlc, Fout_UDPGlcNAc,
     Fout_GDPMan, Fout_GDPFuc, Fout_CMPNeu5Ac,
     r1_f, r2_f, r2_bf, r3_f, r4_f, r5_f, r6_f, r6_gal, r7_f,
     r7_sink, r1_sink, r6_sink,
     r1_urd, r2_urd, r4_urd, r6_urd) = model_params(state, ng=1)

    dXv = Xv * ((mu - mu_d) * V - Fin) / V
    dmAb = (QmAb * V * Xv - Fin * mAb) / V

    dGal = (Fin * (Gal_feed - Gal) + Qgal * V * Xv) / V
    dUrd = (Fin * (Urd_feed - Urd) + Qurd * V * Xv) / V
    dGlc = (Fin * (GLC_FEED - Glc) + Qglc * V * Xv) / V
    dAmm = (Fin * (AMM_FEED - Amm) + Qamm * V * Xv) / V
    dGln = (Fin * (GLN_FEED - Gln) + Qgln * V * Xv) / V
    dLac = (Fin * (LAC_FEED - Lac) + Qlac * V * Xv) / V
    dAsn = (Fin * (ASN_FEED - Asn) + Qasn * V * Xv) / V
    dGlu = (Fin * (GLU_FEED - Glu) + Qglu * V * Xv) / V

    dUDPGal = r6_f + r6_urd + r6_gal - r6_sink - Fout_UDPGal
    dUDPGalNAc = r4_f + r4_urd - Fout_UDPGalNAc
    dUDPGlc = r2_f + r2_bf + r2_urd - Fout_UDPGlc
    dUDPGlcNAc = r1_f + r1_urd - r4_f - r5_f - r1_sink - Fout_UDPGlcNAc
    dGDPMan = r3_f - r7_f - Fout_GDPMan
    dGDPFuc = r7_f - r7_sink - Fout_GDPFuc
    dCMPNeu5Ac = r5_f - Fout_CMPNeu5Ac

    return np.array([
        dXv, dmAb, dGal, dUrd, dGlc, dAmm, dGln, dLac, dAsn, dGlu,
        dUDPGal, dUDPGalNAc, dUDPGlc, dUDPGlcNAc, dGDPMan, dGDPFuc, dCMPNeu5Ac,
    ], dtype="float64")


def model_step(current_state, time, controls, step_len):
    """
    One integration step of the full mechanistic model.

    Parameters
    ----------
    current_state : array-like, shape (17,)
    time : float
    controls : dict or tuple (Fin, Fout, V, Gal_feed, Urd_feed)
    step_len : float

    Returns
    -------
    np.ndarray, shape (17,)
    """
    current_state = _to_1d(current_state)
    step_len = _to_scalar(step_len)

    if isinstance(controls, dict):
        Fin = controls["Fin"]
        Fout_ctrl = controls["Fout"]
        V = controls["V"]
        Gal_feed = controls["Gal_feed"]
        Urd_feed = controls["Urd_feed"]
    else:
        Fin, Fout_ctrl, V, Gal_feed, Urd_feed = controls

    Fin = _to_scalar(Fin)
    Fout_ctrl = _to_scalar(Fout_ctrl)
    V = _to_scalar(V)
    Gal_feed = _to_scalar(Gal_feed)
    Urd_feed = _to_scalar(Urd_feed)

    # Integrate one step with LSODA via odeint — a SINGLE native call that allocates and
    # releases its work arrays each call. This replaces the old pattern of constructing a
    # fresh scipy.integrate.ode(...).integrate() on every call, which leaks native LSODA
    # workspace unboundedly over the millions of calls per EnKF pass and OOM-kills long
    # runs. rtol/atol/mxstep mirror the previous ode+lsoda defaults, so the numerics are
    # unchanged (verified bit-close against a reference trajectory).
    traj = scp.odeint(
        _dynamic_model, current_state, [time, time + step_len],
        args=(Fin, Fout_ctrl, V, Gal_feed, Urd_feed),
        tfirst=True, rtol=1e-6, atol=1e-12, mxstep=3000,
    )
    return _to_1d(traj[-1])


# ─── Open-loop simulation ───────────────────────────────────────────────────

def simulate_dataset(init_state, Fin, Fout, Gal_feed, Urd_feed,
                     V_traj, time_grid, step_len, name=""):
    """
    Run a forward simulation with nominal parameters for one dataset.

    Returns
    -------
    np.ndarray, shape (steps_n, state_num)
    """
    state = _to_1d(init_state)
    traj = []

    for k in tqdm(range(len(Fin)), desc=f"Simulating {name}", leave=False):
        controls_k = {
            "Fin": Fin[k],
            "Fout": Fout[k],
            "V": V_traj[k],
            "Gal_feed": Gal_feed[k],
            "Urd_feed": Urd_feed[k],
        }
        state = model_step(state, time_grid[k], controls_k, step_len[k])
        traj.append(state)

    return np.array(traj)
