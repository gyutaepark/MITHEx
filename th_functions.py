from units import Q_
import numpy as np

def compute_Nu(fluid, Re=None, Pr=None, Pe=None):
    if fluid == "Sodium":
        # Here we use the Dittus-Boelter
        if Re == None or Pr == None:
            raise ValueError("Re and Pr numbers are necessary for Sodium Nu correlation.")
        Nu = 0.023 * Re ** 0.8 * Pr ** 0.3
    elif fluid == "CO2":
        # Here we use Li
        pass
    elif fluid == "Steam":
        # Different Nu correlation, although steam is funky
        pass
    return Nu

def compute_Nu_Dittus_Boelter(Re, Pr, heating=True):
    if heating:
        Nu = 0.023 * Re ** 0.8 * Pr ** 0.4
    else:
        Nu = 0.023 * Re ** 0.8 * Pr ** 0.3
    return Nu

def compute_Nu_Seban_Shimazaki(Pe):
    Nu = 5.0 + 0.025 * Pe ** 0.8
    return Nu

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
    return cp, k, rho, mu

def compute_Na_cp(T):
    T_ = T.m_as('K')
    cp = 1.6582 - 8.4790e-4 * T_ + 4.4541e-7 * T_ ** 2
    return Q_(cp, "J/kg/K")

def compute_Na_k(T):
    T_ = T.m_as('K')
    k = 124.67 - 0.11381 * T_ + 5.5226e-5 * T_ ** 2 - 1.1842e-8 * T_ ** 3
    return Q_(k, "W/m/K")

def compute_Na_rho(T):
    T_ = T.m_as('K')
    T_c = 2503.7
    rho_c = 219.
    f = 275.32
    g = 511.58
    rho = rho_c + f * (1 - T_ / T_c) + g * (1 - T_ / T_c) ** 0.5
    return Q_(rho, "kg / m ** 3")

def compute_Na_mu(T):
    T_ = T.m_as("K")
    mu = np.exp(-6.4406 - 0.3958 * np.log(T_) + 556.835 / T_)
    return Q_(mu, "Pa * s")