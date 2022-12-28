from units import Q_
import numpy as np
import sdf
import scipy.optimize
from CoolProp.CoolProp import PropsSI
from th_functions import fluid_properties, saturated_liquid_properties, saturated_vapor_properties, compute_Nu_Dittus_Boelter, \
    compute_Nu_Seban_Shimazaki, compute_Nu_Chen, compute_k_SS304, compute_k_SS316, compute_NTU_counterflow, compute_friction_factor, \
    compute_equilibrium_quality, compute_flow_quality, compute_Nu_Groeneveld, compute_htc_Dittus_Boelter, compute_two_phase_friction_multiplier

def compute_required_area(inputs):
    # initialize thermal inputs
    Q = Q_(inputs["Thermal Power (MW)"], 'MW').m_as('W')
    L = inputs["Flow Length (m)"]
    primary_fluid = inputs["Primary Fluid"]
    primary_hot = Q_(inputs["Primary Hot Temperature (C)"], 'degC').m_as('K')
    primary_cold = Q_(inputs["Primary Cold Temperature (C)"], 'degC').m_as('K')
    primary_mdot = inputs["Primary Mass Flow Rate (kg/s)"]
    primary_P = Q_(inputs["Primary Pressure (kPa)"], 'kPa').m_as('Pa')
    primary_deltaP = Q_(inputs["Primary Pressure Drop (kPa)"], 'kPa').m_as('Pa')
    secondary_fluid = inputs["Secondary Fluid"]
    secondary_hot = Q_(inputs["Secondary Hot Temperature (C)"], 'degC').m_as('K')
    secondary_cold = Q_(inputs["Secondary Cold Temperature (C)"], 'degC').m_as('K')
    secondary_mdot = inputs["Secondary Mass Flow Rate (kg/s)"]
    secondary_P = Q_(inputs["Secondary Pressure (kPa)"], 'kPa').m_as('Pa')
    secondary_deltaP = Q_(inputs["Secondary Pressure Drop (kPa)"], 'kPa').m_as('Pa')

    # fill out missing thermal parameters
    if np.isnan(primary_mdot.m):
        primary_avg_temperature = (primary_hot + primary_cold) / 2
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
        secondary_avg_temperature = (secondary_hot + secondary_cold) / 2
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
    D = Q_(inputs["Channel Diameter (m)"], 'm')
    flow_area = (np.pi * D ** 2) / 8
    P_wetted = D + np.pi * D / 2
    D_h = 4 * flow_area / P_wetted

    # Compute plate thermal conductivity
    T_avg = (primary_hot + primary_cold + secondary_hot + secondary_cold) / 4
    if plate_material == "SS304":
        plate_k = compute_k_SS304(T_avg)
    elif plate_material == "SS316":
        plate_k = compute_k_SS316(T_avg)
    else:
        raise ValueError("Given plate material of {} is not currently supported.".format(plate_material))

    # Determine flow velocities by finding the limiting pressure drop
    # Both fluids should have an equal fraction of mass flow rate in each channel
    primary_velocity = scipy.optimize.brentq(_compute_velocity, 0.0001, 20., args=(primary_rho, D_h, primary_mu, L, primary_deltaP))
    primary_velocity = Q_(primary_velocity, 'm/s')
    primary_mdot_channel = primary_rho * primary_velocity * flow_area
    secondary_velocity = scipy.optimize.brentq(_compute_velocity, 0.0001, 20., args=(secondary_rho, D_h, secondary_mu, L, secondary_deltaP))
    secondary_velocity = Q_(secondary_velocity, 'm/s')
    secondary_mdot_channel = secondary_rho * secondary_velocity * flow_area

    primary_mdot_ratio = primary_mdot_channel / primary_mdot
    secondary_mdot_ratio = primary_mdot_channel / primary_mdot

    # Calculate heat transfer coefficients
    primary_Re = primary_rho * primary_velocity * D_h / primary_mu
    primary_Pr = primary_mu * primary_cp / primary_k
    # if primary_fluid == "Sodium":
    #     primary_Pe = primary_Re * primary_Pr
    #     print(primary_Pe)
    #     primary_Nu = compute_Nu_Seban_Shimazaki(primary_Pe)
    primary_Nu = compute_Nu_Dittus_Boelter(primary_Re, primary_Pr, heating=False)
    primary_h = primary_Nu * primary_k / D_h

    secondary_Re = secondary_rho * secondary_velocity * D_h / secondary_mu
    secondary_Pr = secondary_mu * secondary_cp / secondary_k
    secondary_Nu = compute_Nu_Dittus_Boelter(secondary_Re, secondary_Pr, heating=True)
    secondary_h = secondary_Nu * secondary_k / D_h

    # Calculate overall heat transfer coefficient
    U = (primary_h ** -1 + plate_thickness * plate_k ** -1 + secondary_h ** -1) ** -1

    # LMTD METHOD
    # Calculate LMTD
    deltaT_1 = primary_hot - secondary_hot
    deltaT_2 = primary_cold - secondary_cold
    LMTD = (deltaT_1 - deltaT_2) / np.log(deltaT_1 / deltaT_2)

    # Gather results
    UA_LMTD = Q / LMTD
    A_LMTD = (UA_LMTD / U)

    # E-NTU METHOD
    # Calculate efficiency
    C_hot = primary_cp * primary_mdot
    C_cold = secondary_cp * secondary_mdot
    C_min = np.minimum(C_hot, C_cold)
    C_max = np.maximum(C_hot, C_cold)
    C_r = C_min / C_max

    q_max = C_min * (primary_hot - secondary_cold)
    e = Q / q_max

    # Use efficiency to calculate NTU
    if C_r == 1:
        NTU = compute_NTU_counterflow(e)
    else:
        NTU = compute_NTU_counterflow(e, C_r)

    # Gather results
    UA_ENTU = NTU * C_min
    A_ENTU = UA_ENTU / U
    
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
    Re = rho * v * D / mu
    f = compute_friction_factor(Re)
    dP = 0.5 * f * rho * v ** 2 * L / D
    return (deltaP - dP)

def fill_out_parameters(Q, c_p, T_hot=None, T_cold=None, m_dot=None):
    """
    This function takes the rated power and specific heat of a fluid,
    along with a required two input parameters and uses those to calculate
    the missing parameter.
    """
    if T_hot == None:
        return Q / (m_dot * c_p) + T_cold
    if T_cold == None:
        return T_hot - Q / (m_dot * c_p)
    if m_dot == None:
        return Q / (c_p * (T_hot - T_cold))
    raise KeyError("Too many inputs. Two optional inputs are required.")

def compute_required_area_SG(inputs):
    np.seterr(all='raise')
    Q = Q_(inputs["Thermal Power (MW)"], 'MW').m_as('W')
    primary_fluid = inputs["Primary Fluid"]
    primary_hot = Q_(inputs["Primary Hot Temperature (C)"], 'degC').m_as('K')
    primary_cold = Q_(inputs["Primary Cold Temperature (C)"], 'degC').m_as('K')
    primary_P = Q_(inputs["Primary Pressure (kPa)"], 'kPa').m_as('Pa')
    primary_deltaP = Q_(inputs["Primary Pressure Drop (kPa)"], 'kPa').m_as('Pa')
    secondary_hot = Q_(inputs["Secondary Hot Temperature (C)"], 'degC').m_as('K')
    secondary_cold = Q_(inputs["Secondary Cold Temperature (C)"], 'degC').m_as('K')
    secondary_P = Q_(inputs["Secondary Pressure (kPa)"], 'kPa').m_as('Pa')
    secondary_deltaP = Q_(inputs["Secondary Pressure Drop (kPa)"], 'kPa').m_as('Pa')

    # Check that necessary parameters are included
    if primary_cold is None:
        raise ValueError("Primary cold temperature must be included as an input")
    elif secondary_cold is None:
        raise ValueError("Secondary cold temperature must be included as an input")

    # Initialize geometry
    plate_thickness = inputs["Plate thickness (m)"]
    plate_material = inputs["Plate material"]
    D = inputs["Channel Diameter (m)"]
    step_size = inputs["Step size (m)"]

    # Import critical heat flux data
    G_scale = sdf.load('2006LUT.sdf', '/G', 'kg/(m2.s)')
    x_scale = sdf.load('2006LUT.sdf', '/x', '1')
    P_scale = sdf.load('2006LUT.sdf', '/P', 'Pa')
    q_crit = sdf.load('2006LUT.sdf', '/q', 'W/m2')

    # Define the forward stepping solution method as an inner function
    # For a given hydraulic diameter, iterate on secondary mass flow rate
    # to converge temperature. Then you return heat transfer per channel,
    # balancing such that the number of channels is the same for 
    flow_area = np.pi * D ** 2 / 8
    perimeter = D + np.pi * D / 2
    D_h = 4 * flow_area / perimeter
    hx_area = perimeter * step_size

    primary_dT = primary_hot - primary_cold
    primary_avg_T = (primary_hot + primary_cold) / 2
    primary_avg_cp, _, primary_avg_rho, primary_avg_mu = fluid_properties(primary_fluid, primary_avg_T, primary_P)
    
    secondary_H_inlet = PropsSI('H', 'P', secondary_P, 'T', secondary_cold, "Water")
    secondary_H_outlet = PropsSI('H', 'P', secondary_P, 'T', secondary_hot, "Water")
    secondary_dH = secondary_H_outlet - secondary_H_inlet

    def find_HX_length(HX_L, searching=True):
        print("starting HX length iteration with channel length {} m".format(HX_L))
        # Set bounds for primary velocity based on Reynolds number, if the maximum velocity has a lower
        # pressure drop than specified, use the maximum velocity
        if primary_fluid in ["Sodium", "Organic"]:
            Re_low = 10.
            Re_high = 20000.
        else:
            Re_low = 1000.
            Re_high = 100000.
        v_low = Re_low * primary_avg_mu / (primary_avg_rho * D_h)
        v_high = Re_high * primary_avg_mu / (primary_avg_rho * D_h)
        try:
            primary_velocity = scipy.optimize.brentq(_compute_velocity, v_low, v_high, args=(primary_avg_rho, D_h, primary_avg_mu, HX_L, primary_deltaP))
        except:
            primary_velocity = v_high
        primary_mdot_channel = primary_velocity * primary_avg_rho * flow_area
        print("Primary velocity: {}, Primary mass flow rate: {}".format(primary_velocity, primary_mdot_channel))

        fluid_sat = PropsSI('H', 'P', secondary_P, 'Q', 0, "Water")
        vapor_sat = PropsSI('H', 'P', secondary_P, 'Q', 1, "Water")
        H_fg = vapor_sat - fluid_sat

        def compute_single_channel(secondary_mdot_channel, searching=True):
            print("Starting single channel iteration with mass flow rates \nSecondary: {} kg/s\nPrimary: {} kg/s".format(secondary_mdot_channel, primary_mdot_channel))
            loc = 0
            dP = 0
            primary_T = primary_cold
            secondary_H = secondary_H_inlet
            secondary_G = secondary_mdot_channel / flow_area
            
            x_e_departure = None
            boiling = False
            bubble_departure = False
            while not boiling and loc < HX_L:
                Q_step, dP_step, boiling = _compute_subcooled_step(
                    D_h,
                    hx_area,
                    flow_area,
                    step_size,
                    primary_fluid,
                    primary_T,
                    primary_P,
                    primary_mdot_channel,
                    secondary_H,
                    secondary_P,
                    secondary_mdot_channel,
                    plate_material,
                    plate_thickness
                )
                dP += dP_step
                loc += step_size
                secondary_H = secondary_H + Q_step / secondary_mdot_channel
                primary_cp, _, _, _ = fluid_properties(primary_fluid, primary_T, primary_P)
                primary_T = primary_T + Q_step / (primary_mdot_channel * primary_cp)
            # The current model cannot converge wall temperature in the initial nucleate boiling
            # step. Adding another single phase step prevents this issue.
            # _compute_single_phase_step()

            # At this point, the evaporator section will be modeled. Each step of the evaporator
            # section before critical heat flux must be solved iteratively as heat transfer coefficients
            # are based on wall temperature and wall temperature is based on heat transfer coefficients
            x_e = (secondary_H - fluid_sat) / H_fg
            DNB = False
            while loc < HX_L and not DNB:
                # Use the correct wall temperature to find heat transfer and step forward
                Q_step, dP_step, x_e_departure = _compute_nucleate_boiling_step(
                    D_h,
                    hx_area,
                    flow_area,
                    step_size,
                    primary_fluid,
                    primary_T,
                    primary_P,
                    primary_mdot_channel,
                    secondary_H,
                    secondary_P,
                    secondary_mdot_channel,
                    plate_material,
                    plate_thickness,
                    x_e_departure
                )
                dP += dP_step
                loc += step_size
                secondary_H = secondary_H + Q_step / secondary_mdot_channel
                x_e = (secondary_H - fluid_sat) / H_fg
                primary_cp, _, _, _ = fluid_properties(primary_fluid, primary_T, primary_P)
                primary_T = primary_T + Q_step / (primary_mdot_channel * primary_cp)
                if x_e > 0:
                    CHF = ((.008 / D_h) ** 0.5) * interpolate_CHF(G_scale, secondary_G, x_scale, x_e, P_scale, secondary_P, q_crit)
                    DNB = Q_step / hx_area > CHF
                print("Boiling", primary_T, secondary_H)

            # At this point water has reached the superheated vapor state and boiling is no longer occurring
            # Iterate through the superheated region until right before the end of the heat exchanger
            while loc < HX_L:
                Q_step, dP_step = _compute_superheated_step(
                    D_h,
                    hx_area,
                    flow_area,
                    step_size,
                    primary_fluid,
                    primary_T,
                    primary_P,
                    primary_mdot_channel,
                    secondary_H,
                    secondary_P,
                    secondary_mdot_channel,
                    plate_material,
                    plate_thickness,
                    x_e_departure
                )
                dP += dP_step
                loc += step_size
                secondary_H = secondary_H + Q_step / secondary_mdot_channel
                x_e = (secondary_H - fluid_sat) / H_fg
                primary_cp, _, _, _ = fluid_properties(primary_fluid, primary_T, primary_P)
                primary_T = primary_T + Q_step / (primary_mdot_channel * primary_cp)
                print("Superheated", primary_T, secondary_H)

            if searching:
                return dP - secondary_deltaP
            else:
                return secondary_H

        # Set bounds for the steam side mass flow rate and iterate to find the maximum mass flow rate before exceeding pressure drop
        _, _, _, mu = fluid_properties("Water", secondary_cold, secondary_P)
        Re_low = 1000
        m_dot_low = 0.25 * Re_low * np.pi * D_h * mu

        Q_primary = primary_mdot_channel * primary_avg_cp * primary_dT
        m_dot_high = Q_primary / secondary_dH
        try:
            secondary_mdot_channel = scipy.optimize.brentq(compute_single_channel, m_dot_low, m_dot_high, rtol=1e-3)
        except ValueError:
            secondary_mdot_channel = m_dot_high

        Q_secondary = secondary_mdot_channel * secondary_dH

        if Q_primary > Q_secondary:
            # Secondary side pressure drop is limiting, reduce the primary mass flow rate to match
            primary_mdot_channel = Q_secondary / (primary_avg_cp * primary_dT)

        outlet_enthalpy = compute_single_channel(secondary_mdot_channel, searching=False)
        print("Iteration error:", outlet_enthalpy / secondary_H_outlet)
        if searching:
            return outlet_enthalpy - secondary_H_outlet
        else:
            return primary_mdot_channel, secondary_mdot_channel
    HX_length = scipy.optimize.brentq(find_HX_length, 0.1, 2, rtol=1e-5)
    primary_mdot_channel, secondary_mdot_channel = find_HX_length(HX_length, searching=False)

    fluid_sat = PropsSI('H', 'P', secondary_P, 'Q', 0, "Water")
    vapor_sat = PropsSI('H', 'P', secondary_P, 'Q', 1, "Water")
    H_fg = vapor_sat - fluid_sat

    loc = 0
    Q_channel = 0
    dP = 0
    primary_T = primary_cold
    secondary_H = secondary_H_inlet
    secondary_G = secondary_mdot_channel / flow_area
    
    x_e_departure = None
    boiling = False
    while not boiling and loc < HX_length:
        Q_step, dP_step, boiling = _compute_subcooled_step(
            D_h,
            hx_area,
            flow_area,
            step_size,
            primary_fluid,
            primary_T,
            primary_P,
            primary_mdot_channel,
            secondary_H,
            secondary_P,
            secondary_mdot_channel,
            plate_material,
            plate_thickness
        )
        dP += dP_step
        Q_channel += Q_step
        loc += step_size
        secondary_H = secondary_H + Q_step / secondary_mdot_channel
        primary_cp, _, _, _ = fluid_properties(primary_fluid, primary_T, primary_P)
        primary_T = primary_T + Q_step / (primary_mdot_channel * primary_cp)
        print("Subcooled", primary_T, secondary_H)

    # At this point, the evaporator section will be modeled. Each step of the evaporator
    # section before critical heat flux must be solved iteratively as heat transfer coefficients
    # are based on wall temperature and wall temperature is based on heat transfer coefficients
    x_e = (secondary_H - fluid_sat) / H_fg
    DNB = False
    while loc < HX_length and not DNB:
        # Use the correct wall temperature to find heat transfer and step forward
        Q_step, dP_step, x_e_departure = _compute_nucleate_boiling_step(
            D_h,
            hx_area,
            flow_area,
            step_size,
            primary_fluid,
            primary_T,
            primary_P,
            primary_mdot_channel,
            secondary_H,
            secondary_P,
            secondary_mdot_channel,
            plate_material,
            plate_thickness,
            x_e_departure
        )
        dP += dP_step
        Q_channel += Q_step
        loc += step_size
        secondary_H = secondary_H + Q_step / secondary_mdot_channel
        x_e = (secondary_H - fluid_sat) / H_fg
        primary_cp, _, _, _ = fluid_properties(primary_fluid, primary_T, primary_P)
        primary_T = primary_T + Q_step / (primary_mdot_channel * primary_cp)
        if x_e > 0:
            CHF = ((.008 / D_h) ** 0.5) * interpolate_CHF(G_scale, secondary_G, x_scale, x_e, P_scale, secondary_P, q_crit)
            DNB = Q_step / hx_area > CHF
        print("Boiling", primary_T, secondary_H)

    # At this point water has reached the superheated vapor state and boiling is no longer occurring
    # Iterate through the superheated region until right before the end of the heat exchanger
    while loc < HX_length:
        Q_step, dP_step = _compute_superheated_step(
            D_h,
            hx_area,
            flow_area,
            step_size,
            primary_fluid,
            primary_T,
            primary_P,
            primary_mdot_channel,
            secondary_H,
            secondary_P,
            secondary_mdot_channel,
            plate_material,
            plate_thickness,
            x_e_departure
        )
        Q_channel += Q_step
        dP += dP_step
        loc += step_size
        secondary_H = secondary_H + Q_step / secondary_mdot_channel
        x_e = (secondary_H - fluid_sat) / H_fg
        primary_cp, _, _, _ = fluid_properties(primary_fluid, primary_T, primary_P)
        primary_T = primary_T + Q_step / (primary_mdot_channel * primary_cp)
        print("Superheated", primary_T, secondary_H)

    num_channels = int(np.ceil(Q / Q_channel))
    primary_mdot = num_channels * primary_mdot_channel
    secondary_mdot = num_channels * secondary_mdot_channel

    results = {}
    results["Primary Channel Mass Flow Rate (kg/s)"] = primary_mdot_channel
    results["Primary Mass Flow Rate (kg/s)"] = primary_mdot
    results["Secondary Channel Mass Flow Rate (kg/s)"] = secondary_mdot_channel
    results["Secondary Mass Flow Rate (kg/s)"] = secondary_mdot
    results["Secondary Pressure Drop (kPa)"] = dP / 1000
    results["Number of Channels"] = num_channels
    results["Heat Exchanger Length"] = HX_length

    return results


def _compute_subcooled_step(D_h, A, A_flow, L, primary_fluid, primary_T, primary_P, primary_mdot, secondary_H, secondary_P, secondary_mdot, plate_material, plate_thickness):    
    boiling = False
    
    primary_cp, primary_k, primary_rho, primary_mu = fluid_properties(primary_fluid, primary_T, primary_P)
    primary_velocity = primary_mdot / (primary_rho * A_flow)
    primary_Re = primary_rho * primary_velocity * D_h / primary_mu
    primary_Pr = primary_mu * primary_cp / primary_k
    if primary_fluid == "Sodium":
        primary_Pe = primary_Re * primary_Pr
        primary_Nu = compute_Nu_Seban_Shimazaki(primary_Pe)
    else:
        primary_Nu = compute_Nu_Dittus_Boelter(primary_Re, primary_Pr, heating=False)
    primary_htc = primary_Nu * primary_k / D_h
    primary_R = (primary_htc * A) ** -1

    secondary_T = PropsSI('T', 'H', secondary_H, 'P', secondary_P, 'Water')
    T_sat = PropsSI('T', 'P', secondary_P, 'Q', 0, 'Water')

    T_avg = (primary_T + secondary_T) / 2
    if plate_material == "SS304":
        plate_k = compute_k_SS304(T_avg)
    elif plate_material == "SS316":
        plate_k = compute_k_SS316(T_avg)
    plate_R = plate_thickness / (plate_k * A)

    # Use a Dittus Boelter equation for single phase convection
    secondary_cp, secondary_k, secondary_rho, secondary_mu = fluid_properties('Water', secondary_T, secondary_P)
    _, _, rho_g, _ = saturated_vapor_properties('Water', secondary_P)
    secondary_velocity = secondary_mdot / (secondary_rho * A_flow)
    secondary_Re = secondary_rho * secondary_velocity * D_h / secondary_mu
    secondary_Pr = secondary_mu * secondary_cp / secondary_k
    secondary_Nu = compute_Nu_Dittus_Boelter(secondary_Re, secondary_Pr, heating=True)
    secondary_htc = secondary_Nu * secondary_k / D_h
    secondary_R = (secondary_htc * A) ** -1

    # Compute flow resistances and heat transfer
    R = primary_R + plate_R + secondary_R
    Q = (primary_T - secondary_T) / R

    # Check onset of subcooled boiling
    T_wall = secondary_T + (primary_T - secondary_T) * secondary_R / R
    q_prime = Q / A
    surface_tension = PropsSI('I', 'P', secondary_P, 'Q', 0, 'Water')
    h_fg = PropsSI('H', 'P', secondary_P, 'Q', 1, 'Water') - PropsSI('H', 'P', secondary_P, 'Q', 0, 'Water')
    criteria = np.sqrt(8 * surface_tension * T_sat * q_prime / (secondary_k * h_fg * rho_g))

    # If subcooled boiling is occurring at this node, the heat transfer must be recalculated for
    # the appropriate regime
    if (T_wall - T_sat) > criteria:
        boiling = True

    f = compute_friction_factor(secondary_Re)
    dP = 0.5 * f * secondary_rho * secondary_velocity ** 2 * L / D_h

    return Q, dP, boiling


def _compute_nucleate_boiling_step(D_h, A, A_flow, L, primary_fluid, primary_T, primary_P, primary_mdot, secondary_H, secondary_P, secondary_mdot, plate_material, plate_thickness, x_e_departure):
    primary_cp, primary_k, primary_rho, primary_mu = fluid_properties(primary_fluid, primary_T, primary_P)
    primary_velocity = primary_mdot / (primary_rho * A_flow)
    primary_Re = primary_rho * primary_velocity * D_h / primary_mu
    primary_Pr = primary_mu * primary_cp / primary_k
    if primary_fluid == "Sodium":
        primary_Pe = primary_Re * primary_Pr
        primary_Nu = compute_Nu_Seban_Shimazaki(primary_Pe)
    else:
        primary_Nu = compute_Nu_Dittus_Boelter(primary_Re, primary_Pr, heating=False)
    primary_htc = primary_Nu * primary_k / D_h
    primary_R = (primary_htc * A) ** -1

    secondary_T = PropsSI('T', 'H', secondary_H, 'P', secondary_P, 'Water')
    T_sat = PropsSI('T', 'P', secondary_P, 'Q', 0, 'Water')

    T_avg = (primary_T + secondary_T) / 2
    if plate_material == "SS304":
        plate_k = compute_k_SS304(T_avg)
    elif plate_material == "SS316":
        plate_k = compute_k_SS316(T_avg)
    plate_R = plate_thickness / (plate_k * A)
    
    secondary_G = secondary_mdot / A_flow
    cp_l, k_l, rho_l, mu_l = saturated_liquid_properties('Water', secondary_P)
    _, _, rho_g, mu_g = saturated_vapor_properties('Water', secondary_P)
    x_e = compute_equilibrium_quality(secondary_H, secondary_P)
    if x_e_departure is None:
        x = 0
    else:
        x = compute_flow_quality(x_e, x_e_departure)
    secondary_Re = secondary_G * D_h * (1 - x) / mu_l
    Pr_l = mu_l * cp_l / k_l

    X_tt = (x / (1 - x)) ** 0.9 \
        * (rho_l / rho_g) ** 0.5 \
        * (mu_g / mu_l) ** 0.1
    if X_tt > 0.1:
        F = (2.35 * (0.213 + X_tt) ** 0.736)
    else:
        F = 1

    htc_c = F * compute_htc_Dittus_Boelter(secondary_Re, Pr_l, k_l, D_h)

    # Since the nucleate boiling heat transfer coefficient is dependent on wall temperature,
    # an iterative solution is required.
    def _converge_nucleate_boiling(T_wall):
        sigma = PropsSI('I', 'P', secondary_P, 'Q', 0, 'Water')
        h_fg = PropsSI('H', 'P', secondary_P, 'Q', 1, 'Water') - PropsSI('H', 'P', secondary_P, 'Q', 0, 'Water')
        htc_nb = compute_Nu_Chen(secondary_Re, k_l, cp_l, rho_l, sigma, mu_l, h_fg, rho_g, T_wall, T_sat, F)
        htc = htc_c + htc_nb

        secondary_R = (htc * A) ** -1

        Q_primary = (primary_T - T_wall) / (primary_R + plate_R)
        Q_secondary = (T_wall - T_sat) / secondary_R

        return Q_primary - Q_secondary

    T_wall = scipy.optimize.brentq(_converge_nucleate_boiling, primary_T, T_sat)
    sigma = PropsSI('I', 'P', secondary_P, 'Q', 0, 'Water')
    h_fg = PropsSI('H', 'P', secondary_P, 'Q', 1, 'Water') - PropsSI('H', 'P', secondary_P, 'Q', 0, 'Water')
    htc_nb = compute_Nu_Chen(secondary_Re, k_l, cp_l, rho_l, sigma, mu_l, h_fg, rho_g, T_wall, T_sat, F)
    htc = htc_c + htc_nb

    secondary_R = (htc * A) ** -1
    Q = (T_wall - T_sat) / secondary_R

    secondary_H = secondary_H + (Q / secondary_mdot)
    if x_e_departure is None:
        # Before bubble departure pressure drop is modeled as single phase
        f = compute_friction_factor(secondary_Re)
        secondary_velocity = secondary_mdot / (rho_l * A_flow)
        dP = 0.5 * f * rho_l * secondary_velocity ** 2 * D_h / L

        # Check for bubble departure, indicating that flow quality is no longer 0
        # Saha-Zuber correlation 13.15
        secondary_Pe = secondary_Re * Pr_l
        secondary_T = PropsSI('T', 'P', secondary_P, 'H', secondary_H, 'Water')
        q_prime = Q / A
        if secondary_Pe < 7e4 and secondary_T > T_sat - 0.0022 * q_prime * D_h / k_l:
            x_e_departure = x_e
        elif secondary_T > T_sat - 154. * q_prime / (secondary_G * cp_l):
            x_e_departure = x_e
    else:
        v_lo = secondary_mdot / (rho_l * A_flow)
        Re_lo = rho_l * v_lo * D_h / mu_l
        f_lo = compute_friction_factor(Re_lo)

        v_go = secondary_mdot / (rho_l * A_flow)
        Re_go = rho_l * v_lo * D_h / mu_l
        f_go = compute_friction_factor(Re_lo)

        rho_m = (x / rho_g + (1 - x) / rho_l)

        friction_multiplier = compute_two_phase_friction_multiplier(
            x,
            secondary_G,
            D_h,
            sigma,
            rho_l,
            rho_g,
            rho_m,
            mu_l,
            mu_g,
            f_lo,
            f_go
        )
        dP = friction_multiplier * 0.5 * f_lo * rho_l * v_lo ** 2 * L / D_h
    return Q, dP, x_e_departure

def _compute_superheated_step(D_h, A, A_flow, L, primary_fluid, primary_T, primary_P, primary_mdot, secondary_H, secondary_P, secondary_mdot, plate_material, plate_thickness, x_e_departure):
    primary_cp, primary_k, primary_rho, primary_mu = fluid_properties(primary_fluid, primary_T, primary_P)
    primary_velocity = primary_mdot / (primary_rho * A_flow)
    primary_Re = primary_rho * primary_velocity * D_h / primary_mu
    primary_Pr = primary_mu * primary_cp / primary_k
    if primary_fluid == "Sodium":
        primary_Pe = primary_Re * primary_Pr
        primary_Nu = compute_Nu_Seban_Shimazaki(primary_Pe)
    else:
        primary_Nu = compute_Nu_Dittus_Boelter(primary_Re, primary_Pr, heating=False)
    primary_htc = primary_Nu * primary_k / D_h
    primary_R = (primary_htc * A) ** -1

    secondary_T = PropsSI('T', 'H', secondary_H, 'P', secondary_P, 'Water')

    T_avg = (primary_T + secondary_T) / 2
    if plate_material == "SS304":
        plate_k = compute_k_SS304(T_avg)
    elif plate_material == "SS316":
        plate_k = compute_k_SS316(T_avg)
    plate_R = plate_thickness / (plate_k * A)

    # Use a Dittus Boelter equation for single phase convection
    H_g = PropsSI('H', 'P', secondary_P, 'Q', 1, "Water")
    if secondary_H > H_g:
        secondary_cp, secondary_k, secondary_rho, secondary_mu = fluid_properties('Water', secondary_T, secondary_P)
    else:
        secondary_cp, secondary_k, secondary_rho, secondary_mu = saturated_vapor_properties('Water', secondary_P)
    _, _, rho_g, mu_g = saturated_vapor_properties('Water', secondary_P)
    secondary_velocity = secondary_mdot / (secondary_rho * A_flow)
    secondary_Re = secondary_rho * secondary_velocity * D_h / secondary_mu
    secondary_Pr = secondary_mu * secondary_cp / secondary_k
    secondary_Nu = compute_Nu_Dittus_Boelter(secondary_Re, secondary_Pr, heating=True)
    secondary_htc = secondary_Nu * secondary_k / D_h

    # Compute flow resistances and heat transfer
    secondary_R = (secondary_htc * A) ** -1
    R = primary_R + plate_R + secondary_R
    Q = (primary_T - secondary_T) / R

    # Compute pressure drop
    if secondary_H > H_g:
        f = compute_friction_factor(secondary_Re)
        dP = 0.5 * f * secondary_rho * secondary_velocity ** 2 * L / D_h
    else:
        _, _, rho_l, mu_l = saturated_liquid_properties('Water', secondary_P)
        H_f = PropsSI('H', 'P', secondary_P, 'Q', 0, "Water")
        x_e = (secondary_H - H_f) / (H_g - H_f)
        x = compute_flow_quality(x_e, x_e_departure)
        secondary_G = secondary_mdot / A_flow
        sigma = PropsSI('I', 'P', secondary_P, 'Q', 0, 'Water')

        v_lo = secondary_mdot / (rho_l * A_flow)
        Re_lo = rho_l * v_lo * D_h / mu_l
        f_lo = compute_friction_factor(Re_lo)

        v_go = secondary_mdot / (rho_g * A_flow)
        Re_go = rho_g * v_go * D_h / mu_g
        f_go = compute_friction_factor(Re_go)

        rho_m = (x / rho_g + (1 - x) / rho_l)

        friction_multiplier = compute_two_phase_friction_multiplier(
            x,
            secondary_G,
            D_h,
            sigma,
            rho_l,
            rho_g,
            rho_m,
            mu_l,
            mu_g,
            f_lo,
            f_go
        )
        dP = friction_multiplier * 0.5 * f_lo * rho_l * v_lo ** 2 * L / D_h

    return Q, dP

def interpolate_CHF(G_scale, G, x_scale, x, P_scale, P, q_crit):
    """ Determine critical heat flux of nucleate boiling flow by interpolating
        the 2006 CHF look-up tables by Groeneveld.
    """
    # Remove units
    G_0 = None
    G_1 = None
    x_0 = None
    x_1 = None
    P_0 = None
    P_1 = None

    # Find the nearest values in the scales
    for iter in range(G_scale.data.shape[0]):
        if G < G_scale.data[iter]:
            G_1 = G_scale.data[iter]
            G_0 = G_scale.data[iter-1]
            G_index = iter
            break

    for iter in range(x_scale.data.shape[0]):
        if x < x_scale.data[iter]:
            x_1 = x_scale.data[iter]
            x_0 = x_scale.data[iter-1]
            x_index = iter
            break

    for iter in range(P_scale.data.shape[0]):
        if P < P_scale.data[iter]:
            P_1 = P_scale.data[iter]
            P_0 = P_scale.data[iter-1]
            P_index = iter
            break

    # Calculate interpolation parameters
    G_d = (G - G_0) / (G_1 - G_0)
    x_d = (x - x_0) / (x_1 - x_0)
    P_d = (P - P_0) / (P_1 - P_0)

    # Fill out the 3 dimensional space
    q_000 = q_crit.data[G_index-1][x_index-1][P_index-1]
    q_100 = q_crit.data[G_index][x_index-1][P_index-1]
    q_001 = q_crit.data[G_index-1][x_index-1][P_index]
    q_101 = q_crit.data[G_index][x_index-1][P_index]
    q_010 = q_crit.data[G_index-1][x_index][P_index-1]
    q_110 = q_crit.data[G_index][x_index][P_index-1]
    q_011 = q_crit.data[G_index-1][x_index][P_index]
    q_111 = q_crit.data[G_index][x_index][P_index]

    # Interpolate along G
    q_00 = q_000 * (1 - G_d) + q_100 * G_d
    q_01 = q_001 * (1 - G_d) + q_101 * G_d
    q_10 = q_010 * (1 - G_d) + q_110 * G_d
    q_11 = q_011 * (1 - G_d) + q_111 * G_d

    # Interpolate along x
    q_0 = q_00 * (1 - x_d) + q_10 * x_d
    q_1 = q_01 * (1 - x_d) + q_11 * x_d

    # Interpolate along P
    q = q_0 * (1 - P_d) + q_1 * P_d

    return q