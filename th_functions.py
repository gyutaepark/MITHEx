from units import Q_
import numpy as np
from CoolProp.CoolProp import PropsSI

def compute_NTU_counterflow(e, C_r=None):
    if C_r is None:
        NTU = e * (1 - e) ** -1
    else:
        NTU = ((C_r - 1) ** -1) * np.log((e - 1) * (e * C_r - 1) ** -1)
    return NTU

def compute_Nu_Dittus_Boelter(Re, Pr, heating=True):
    if heating:
        Nu = 0.023 * Re ** 0.8 * Pr ** 0.4
    else:
        Nu = 0.023 * Re ** 0.8 * Pr ** 0.3
    return Nu

def compute_htc_Dittus_Boelter(Re, Pr, k, D, heating=True):
    if heating:
        htc = (k * D ** -1) * 0.023 * Re ** 0.8 * Pr ** 0.4
    else:
        htc = (k * D ** -1) * 0.023 * Re ** 0.8 * Pr ** 0.3
    return htc

def compute_Nu_Seban_Shimazaki(Pe):
    Nu = 5.0 + 0.025 * Pe ** 0.8
    return Nu

def compute_Nu_Chen(Re, k_f, cp_f, rho_f, sigma, mu_f, h_fg, rho_g, T_w, T_sat, F):
    if T_w > 647:
        # Saturation pressure must be found below critical point
        T_w = 647
    p_sat_w = PropsSI('P', 'T', T_w, 'Q', 0, 'Water')
    p_sat = PropsSI('P', 'T', T_sat, 'Q', 0, 'Water')

    S = (1 + 2.53e-6 * (F * 1.25 * Re) ** 1.17) ** -1
    htc = S * 0.00122 * (
        k_f ** 0.79 * cp_f ** 0.45 * rho_f ** 0.49 * (
            sigma ** 0.5 * mu_f ** 0.29 * h_fg ** 0.24 * rho_g ** 0.24
        ) ** -1
    ) * (T_w - T_sat) ** 0.24 * (p_sat_w - p_sat) ** 0.75
    return htc

def compute_Nu_Groeneveld(Re_g, Pr_g, rho_g, rho_f, x):
    """ Computes the nusselt number in film boiling conditions according to the Groeneveld
        correlation.
    """
    Y = (
        1 - 0.1 * ((rho_f / rho_g - 1) ** 0.4) * ((1 - x) ** 0.4)
    ) * (
        x + rho_f / rho_g * (1 - x)
    ) ** -1.06
    return (
        0.052 * (Re_g * (x + (1 - x) * rho_g / rho_f)) ** 0.688 * (Pr_g ** 1.26) * Y
    )

def fluid_properties(fluid, T, P):
    """
    Returns the properties of given fluid at temperature T and pressure P.
    Returns in the order:
        cp, k, rho, mu
    """
    if fluid == "Sodium":
        cp = compute_Na_cp(T)
        k = compute_Na_k(T)
        rho = compute_Na_rho(T)
        mu = compute_Na_mu(T)
    elif fluid == "Organic":
        cp = 2500
        k = 0.11
        rho = 813.
        mu = 2.3e-4
    else:
        cp = PropsSI('C', 'T', T, 'P', P, fluid)
        k = PropsSI('L', 'T', T, 'P', P, fluid)
        rho = PropsSI('D', 'T', T, 'P', P, fluid)
        mu = PropsSI('V', 'T', T, 'P', P, fluid)
    return cp, k, rho, mu

def saturated_liquid_properties(fluid, P):
    """
    Returns the properties of saturated liquid at pressure P.
    Returns in the order:
        cp, k, rho, mu
    """
    cp = PropsSI('C', 'Q', 0, 'P', P, fluid)
    k = PropsSI('L', 'Q', 0, 'P', P, fluid)
    rho = PropsSI('D', 'Q', 0, 'P', P, fluid)
    mu = PropsSI('V', 'Q', 0, 'P', P, fluid)
    return cp, k, rho, mu

def saturated_vapor_properties(fluid, P):
    cp = PropsSI('C', 'Q', 1, 'P', P, fluid)
    k = PropsSI('L', 'Q', 1, 'P', P, fluid)
    rho = PropsSI('D', 'Q', 1, 'P', P, fluid)
    mu = PropsSI('V', 'Q', 1, 'P', P, fluid)
    return cp, k, rho, mu

def compute_equilibrium_quality(h, P):
    h_f = PropsSI("H", "P", P, "Q", 0, "Water")
    h_fg = PropsSI("H", "P", P, "Q", 1, "Water") - h_f
    x_e = (h - h_f) * h_fg ** -1
    return x_e

def compute_flow_quality(x_e, x_e_departure):
    x = x_e - x_e_departure * np.exp((x_e * x_e_departure ** -1) - 1)
    return x

def compute_Na_cp(T):
    cp = 1.6582 - 8.4790e-4 * T + 4.4541e-7 * T ** 2
    return cp * 1000

def compute_Na_k(T):
    k = 124.67 - 0.11381 * T + 5.5226e-5 * T ** 2 - 1.1842e-8 * T ** 3
    return k

def compute_Na_rho(T):
    T_c = 2503.7
    rho_c = 219.
    f = 275.32
    g = 511.58
    rho = rho_c + f * (1 - T / T_c) + g * (1 - T / T_c) ** 0.5
    return rho

def compute_Na_mu(T):
    mu = np.exp(-6.4406 - 0.3958 * np.log(T) + 556.835 / T)
    return mu

def compute_k_SS304(T):
    k = 8.16 + 1.618e-2 * T
    return k

def compute_k_SS316(T):
    k = 12.41 + 3.279e-3 * T
    return k

def compute_friction_factor(Re):
    if Re < 3000:
        f = 64/Re
    else:
        f = 0.316 * Re ** -0.25
    return f

def compute_two_phase_friction_multiplier(x, G, D_h, sigma, rho_l, rho_g, rho_m, mu_l, mu_g, f_lo, f_vo):
    """ Calculates the two phase friction factor multiplier based on the
        Friedel correlation

        Correlation is presented in Todreas and Kazimi, Nuclear Systems
            Vol 1, 2nd ed, section 11.6.3.4

        x is flow quality
        f_lo is liquid only friction factor
        f_vo is vapor only friction factor
    """
    g = 9.81
    E = ((1 - x) ** 2 + x ** 2 * rho_l * f_vo * (rho_g * f_lo) ** -1)
    F = (x ** 0.78 * (1 - x) ** 0.224)
    H = ((rho_l * rho_g ** -1) ** 0.91 * (mu_g * mu_l ** -1) ** 0.19 * (1 - mu_g * mu_l ** -1) ** 0.7)
    Fr = (G ** 2 * (g * D_h * rho_m ** 2) ** -1)
    We = (G ** 2 * D_h * (sigma * rho_m) ** -1)
    phi_lo = E + 3.24 * F * H * (Fr ** 0.0454 * We ** 0.035) ** -1
    return phi_lo