from units import Q_
import numpy as np
import scipy.optimize
from th_functions import fluid_properties, compute_Nu_Dittus_Boelter, \
    compute_Nu_Seban_Shimazaki, compute_k_SS304, compute_k_SS316, compute_NTU_counterflow, \
    compute_friction_factor

def compute_required_area(inputs):
    # initialize thermal inputs
    Q = Q_(inputs["Thermal Power (MW)"], 'MW')
    L = Q_(inputs["Flow Length (m)"], 'm')
    primary_fluid = inputs["Primary Fluid"]
    primary_hot = Q_(inputs["Primary Hot Temperature (C)"], 'degC').to('K')
    primary_cold = Q_(inputs["Primary Cold Temperature (C)"], 'degC').to('K')
    primary_mdot = Q_(inputs["Primary Mass Flow Rate (kg/s)"], 'kg/s')
    primary_P = Q_(inputs["Primary Pressure (kPa)"], 'kPa')
    primary_deltaP = Q_(inputs["Primary Pressure Drop (kPa)"], 'kPa')
    secondary_fluid = inputs["Secondary Fluid"]
    secondary_hot = Q_(inputs["Secondary Hot Temperature (C)"], 'degC').to('K')
    secondary_cold = Q_(inputs["Secondary Cold Temperature (C)"], 'degC').to('K')
    secondary_mdot = Q_(inputs["Secondary Mass Flow Rate (kg/s)"], 'kg/s')
    secondary_P = Q_(inputs["Secondary Pressure (kPa)"], 'kPa')
    secondary_deltaP = Q_(inputs["Secondary Pressure Drop (kPa)"], 'kPa')

    # fill out missing thermal parameters
    if np.isnan(primary_mdot.m):
        primary_avg_temperature = (primary_hot.to("K") + primary_cold.to("K")) / 2
        primary_cp, primary_k, primary_rho, primary_mu = fluid_properties(primary_fluid, primary_avg_temperature, primary_P)
        primary_mdot = fill_out_parameters(Q, primary_cp, T_hot=primary_hot, T_cold=primary_cold)
    elif np.isnan(primary_hot.m):
        primary_cp, primary_k, primary_rho, primary_mu = fluid_properties(primary_fluid, primary_cold, primary_P)
        primary_hot = fill_out_parameters(Q, primary_cp, T_cold=primary_cold, m_dot=primary_mdot)
    elif np.isnan(primary_cold.m):
        primary_cp, primary_k, primary_rho, primary_mu = fluid_properties(primary_fluid, primary_hot, primary_P)
        primary_cold = fill_out_parameters(Q, primary_cp, T_hot=primary_hot, m_dot=primary_mdot)

    # fill out missing thermal parameters
    if np.isnan(secondary_mdot.m):
        secondary_avg_temperature = (secondary_hot.to("K") + secondary_cold.to("K")) / 2
        secondary_cp, secondary_k, secondary_rho, secondary_mu = fluid_properties(secondary_fluid, secondary_avg_temperature, secondary_P)
        secondary_mdot = fill_out_parameters(Q, secondary_cp, T_hot=secondary_hot, T_cold=secondary_cold)
    elif np.isnan(secondary_hot.m):
        secondary_cp, secondary_k, secondary_rho, secondary_mu = fluid_properties(secondary_fluid, secondary_cold, secondary_P)
        secondary_hot = fill_out_parameters(Q, secondary_cp, T_cold=secondary_cold, m_dot=secondary_mdot)
    elif np.isnan(secondary_cold.m):
        secondary_cp, secondary_k, secondary_rho, secondary_mu = fluid_properties(secondary_fluid, secondary_hot, secondary_P)
        secondary_cold = fill_out_parameters(Q, secondary_cp, T_hot=secondary_hot, m_dot=secondary_mdot)

    # Initialize geometry
    plate_thickness = Q_(inputs["Plate thickness (m)"], 'm')
    plate_material = inputs["Plate material"]
    D_h = 2 * Q_(inputs["Plate gap (m)"], 'm')

    # Compute plate thermal conductivity
    T_avg = (primary_hot + primary_cold + secondary_hot + secondary_cold) / 4
    if plate_material == "SS304":
        plate_k = compute_k_SS304(T_avg)
    elif plate_material == "SS316":
        plate_k = compute_k_SS316(T_avg)
    else:
        raise ValueError("Given plate material of {} is not currently supported.".format(plate_material))

    # Determine flow velocities
    primary_velocity = scipy.optimize.brentq(_compute_velocity, 0.0001, 20., args=(primary_rho, D_h, primary_mu, L, primary_deltaP))
    primary_velocity = Q_(primary_velocity, 'm/s')
    secondary_velocity = scipy.optimize.brentq(_compute_velocity, 0.0001, 20., args=(secondary_rho, D_h, secondary_mu, L, secondary_deltaP))
    secondary_velocity = Q_(secondary_velocity, 'm/s')

    # Calculate heat transfer coefficients
    primary_Re = (primary_rho * primary_velocity * D_h * primary_mu ** -1).to('')
    primary_Pr = (primary_mu * primary_cp * primary_k ** -1).to('')
    if primary_fluid == "Sodium":
        primary_Pe = primary_Re * primary_Pr
        primary_Nu = compute_Nu_Seban_Shimazaki(primary_Pe)
    else:
        primary_Nu = compute_Nu_Dittus_Boelter(primary_Re, primary_Pr, heating=False)
    print(primary_k)
    primary_h = primary_Nu * primary_k * D_h ** -1

    secondary_Re = (secondary_rho * secondary_velocity * D_h * secondary_mu ** -1).to('')
    secondary_Pr = (secondary_mu * secondary_cp * secondary_k ** -1).to('')
    secondary_Nu = compute_Nu_Dittus_Boelter(secondary_Re, secondary_Pr, heating=True)
    secondary_h = secondary_Nu * secondary_k * D_h ** -1

    # Calculate overall heat transfer coefficient
    U = (primary_h ** -1 + plate_thickness * plate_k ** -1 + secondary_h ** -1) ** -1

    # LMTD METHOD
    # Calculate LMTD
    deltaT_1 = primary_hot - secondary_hot
    deltaT_2 = primary_cold - secondary_cold
    LMTD = (deltaT_1 - deltaT_2) / np.log(deltaT_1.m_as('K') / deltaT_2.m_as('K'))

    # Gather results
    UA_LMTD = Q / LMTD
    A_LMTD = (UA_LMTD / U).to("m**2")

    # E-NTU METHOD
    # Calculate efficiency
    C_hot = primary_cp * primary_mdot
    C_cold = secondary_cp * secondary_mdot
    C_min = np.minimum(C_hot.m_as('kW / K'), C_cold.m_as('kW / K'))
    C_max = np.maximum(C_hot.m_as('kW / K'), C_cold.m_as('kW / K'))
    C_r = C_min / C_max

    q_max = C_min * (primary_hot.m_as('K') - secondary_cold.m_as('K'))
    e = Q.m_as('kW') / q_max

    # Use efficiency to calculate NTU
    if C_r == 1:
        NTU = compute_NTU_counterflow(e)
    else:
        NTU = compute_NTU_counterflow(e, C_r)

    # Gather results
    UA_ENTU = Q_(NTU * C_min, 'kW / K')
    A_ENTU = (UA_ENTU / U).to("m ** 2")
    
    # Store results to be displayed
    results = {}
    results["Primary velocity"] = primary_velocity
    results["Primary Re"] = primary_Re
    results["Primary Nu"] = primary_Nu
    results["Primary htc"] = primary_h
    results["Secondary velocity"] = secondary_velocity
    results["Secondary Re"] = secondary_Re
    results["Secondary Nu"] = secondary_Nu
    results["Secondary htc"] = secondary_h
    results["Overall htc"] = U
    results["NTU"] = NTU
    results["UA LMTD method"] = UA_LMTD
    results["UA e-NTU method"] = UA_ENTU
    results["Heat transfer area LMTD method"] = A_LMTD
    results["Heat transfer area e-NTU method"] = A_ENTU

    return results

def _compute_velocity(v, rho, D, mu, L, deltaP):
    v_ = Q_(v, 'm/s')
    Re = (rho * v_ * D * mu ** -1).to("")
    f = compute_friction_factor(Re)
    dP = (0.5 * f * rho * v_ ** 2 * L * D ** -1).to('kPa')
    return (deltaP - dP).m_as('kPa')

def fill_out_parameters(Q, c_p, T_hot=None, T_cold=None, m_dot=None):
    """
    This function takes the rated power and specific heat of a fluid,
    along with a required two input parameters and uses those to calculate
    the missing parameter.
    """
    if T_hot == None:
        return (Q / (m_dot * c_p) + T_cold).to('K')
    if T_cold == None:
        return (T_hot - Q / (m_dot * c_p)).to('K')
    if m_dot == None:
        return (Q / (c_p * (T_hot - T_cold))).to('kg / s')
    raise KeyError("Too many inputs. Two optional inputs are required.")
    