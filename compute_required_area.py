from CoolProp.CoolProp import PropsSI
from units import Q_
import numpy as np
from th_functions import fluid_properties, compute_Nu_Dittus_Boelter

def compute_required_area(inputs):
    # initialize thermal inputs
    Q = inputs["Thermal Power"]
    flow = inputs["HX Flow Type"]
    primary_fluid = inputs["Primary Fluid"]
    primary_hot = inputs["Primary Hot Temperature (C)"]
    primary_cold = inputs["Primary Cold Temperature (C)"]
    primary_mdot = inputs["Primary Mass Flow Rate (kg/s)"]
    primary_P = inputs["Primary Pressure (kPa)"]
    secondary_fluid = inputs["Secondary Fluid"]
    secondary_fluid = inputs["Secondary Fluid"]
    secondary_hot = inputs["Secondary Hot Temperature (C)"]
    secondary_cold = inputs["Secondary Cold Temperature (C)"]
    secondary_mdot = inputs["Secondary Mass Flow Rate (kg/s)"]
    secondary_P = inputs["Secondary Pressure (kPa)"]

    # fill out missing thermal parameters
    if primary_mdot == None:
        primary_avg_temperature = (primary_hot.to("K") + primary_cold.to("K")) / 2
        primary_cp, primary_k, primary_rho, primary_mu = fluid_properties(primary_fluid, primary_avg_temperature, primary_P)
        primary_mdot = fill_out_parameters(Q, primary_cp, T_hot=primary_hot, T_cold=primary_cold)
    elif primary_hot == None:
        primary_cp, primary_k, primary_rho, primary_mu = fluid_properties(primary_fluid, primary_cold, primary_P)
        primary_cold = fill_out_parameters(Q, primary_cp, T_cold=primary_cold, m_dot=primary_mdot)
    elif primary_cold == None:
        primary_cp, primary_k, primary_rho, primary_mu = fluid_properties(primary_fluid, primary_hot, primary_P)
        primary_hot = fill_out_parameters(Q, primary_cp, T_hot=primary_hot, m_dot=primary_mdot)
    
    # fill out missing thermal parameters
    if secondary_mdot == None:
        secondary_avg_temperature = (secondary_hot.to("K") + secondary_cold.to("K")) / 2
        secondary_cp, secondary_k, secondary_rho, secondary_mu = fluid_properties(secondary_fluid, secondary_avg_temperature, secondary_P)
        secondary_mdot = fill_out_parameters(Q, secondary_cp, T_hot=secondary_hot, T_cold=secondary_cold)
    elif secondary_hot == None:
        secondary_cp, secondary_k, secondary_rho, secondary_mu = fluid_properties(secondary_fluid, secondary_cold, secondary_P)
        secondary_cold = fill_out_parameters(Q, secondary_cp, T_cold=secondary_cold, m_dot=secondary_mdot)
    elif secondary_cold == None:
        secondary_cp, secondary_k, secondary_rho, secondary_mu = fluid_properties(secondary_fluid, secondary_hot, secondary_P)
        secondary_hot = fill_out_parameters(Q, secondary_cp, T_hot=secondary_hot, m_dot=secondary_mdot)

    # initialize geometry
    tube_outer_diameter = inputs["Tube O.D. (in)"]
    tube_inner_diameter = tube_outer_diameter - inputs["Tube Thickness (in)"]
    num_tubes = inputs["# Tubes"]
    shell_diameter = inputs["Shell Diameter (in)"]
    tube_area = 0.25 * np.pi * tube_inner_diameter ** 2
    total_tube_area = tube_area * num_tubes
    shell_flow_area = 0.25 * np.pi * shell_diameter ** 2 - total_tube_area
    shell_wetted_perimeter = np.pi * (num_tubes * tube_outer_diameter + shell_diameter)

    # calculate characteristic numbers
    # Use Dittus-Boelter for tube side that is not Steam
    if inputs["Shell Fluid"] == "Primary":
        primary_velocity = primary_mdot / (primary_rho * shell_flow_area)
        primary_Re = primary_rho * primary_velocity * tube_inner_diameter / primary_mu
        primary_Pr = primary_mu * primary_cp / primary_k
        primary_Nu
        
        secondary_velocity = secondary_mdot / (secondary_rho * total_tube_area)
        secondary_Re = secondary_rho * secondary_velocity * tube_inner_diameter / secondary_mu
        secondary_Pr = secondary_mu * secondary_cp / secondary_k
        secondary_Nu = compute_Nu_Dittus_Boelter(secondary_Re, secondary_Pr)
        secondary_h = secondary_Nu * secondary_k / tube_inner_diameter

    if flow == "Counter-Flow":
        deltaT_1 = primary_hot - secondary_hot
        deltaT_2 = primary_cold - secondary_cold
    elif flow == "Parallel":
        deltaT_1 = primary_hot - secondary_cold
        deltaT_2 = primary_cold - secondary_hot
    LMTD = (deltaT_1 - deltaT_2) / np.log(deltaT_1 - deltaT_2)

    UA = Q / LMTD

    # Need method of determining shell side and tube side fluid
    Nu_primary = compute_Nu(primary_fluid)
    pass

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
    