from units import Q_
import numpy as np
from CoolProp.CoolProp import PropsSI
from th_functions import fluid_properties
from compute_required_area import fill_out_parameters

def compute_cycle_efficiency(inputs):
    secondary_fluid = inputs["Secondary Fluid"]
    secondary_hot = Q_(inputs["Secondary Hot Temperature (C)"], 'degC').m_as('K')
    secondary_cold = Q_(inputs["Secondary Cold Temperature (C)"], 'degC').m_as('K')
    secondary_min = Q_(inputs["Secondary Minimum Temperature (C)"], 'degC').m_as('K')
    secondary_P_high = Q_(inputs["Secondary Pressure (kPa)"], 'kPa').m_as('Pa')
    secondary_mdot = inputs["Secondary Mass Flow Rate (kg/s)"]
    Q_in = Q_(inputs["Thermal Power (MW)"], 'MW').m_as('W')
    e_pump = inputs["Pump/Compressor Efficiency"]
    e_turbine = inputs["Turbine Efficiency"]
    compression_ratio = inputs["Compression Ratio"]

    # fill out missing thermal parameters
    if np.isnan(secondary_mdot):
        secondary_avg_temperature = (secondary_hot+secondary_cold)/2
        secondary_cp, secondary_k, secondary_rho, secondary_mu = fluid_properties(secondary_fluid, secondary_avg_temperature, secondary_P_high)
        secondary_mdot = fill_out_parameters(Q_in, secondary_cp, T_hot=secondary_hot, T_cold=secondary_cold)
    elif np.isnan(secondary_hot):
        secondary_cp, secondary_k, secondary_rho, secondary_mu = fluid_properties(secondary_fluid, secondary_cold, secondary_P_high)
        secondary_hot = fill_out_parameters(Q_in, secondary_cp, T_cold=secondary_cold, m_dot=secondary_mdot)
    elif np.isnan(secondary_cold):
        secondary_cp, secondary_k, secondary_rho, secondary_mu = fluid_properties(secondary_fluid, secondary_hot, secondary_P_high)
        secondary_cold = fill_out_parameters(Q_in, secondary_cp, T_hot=secondary_hot, m_dot=secondary_mdot)

    # This function is designed to compute the energy lost due to the pump/compressor
    # and the electrical energy generated by the turbine, calculating cycle efficiency
    # by dividing the net power output by the thermal power input.

    if np.isnan(secondary_min):
        # State 2 is defined by the HX cold temperature and operating pressure, and is immediately after exiting the pump/compressor
        P_2 = secondary_P_high
        T_2 = secondary_cold
        h_2 = PropsSI("H", "T", T_2, "P", P_2, secondary_fluid)
        s_2 = PropsSI("S", "T", T_2, "P", P_2, secondary_fluid)

        # Calculate the pump/compressor power
        P_1 = P_2/compression_ratio
        if secondary_fluid == "Water" or secondary_fluid == "CarbonDioxide":
            # Approximate the fluid density with state 2
            rho = PropsSI("D", "T", T_2, "P", P_2, secondary_fluid)
            Q_pump = (secondary_mdot*(P_2-P_1)/rho)/e_pump
        elif secondary_fluid == "Air":
            C_v = PropsSI("O", "T", T_2, "P", P_2, secondary_fluid)
            C_p = PropsSI("C", "T", T_2, "P", P_2, secondary_fluid)
            gamma = C_p/C_v
            T_1 = T_2/(compression_ratio**((gamma-1)/(gamma)))
            h_1 = PropsSI("H", "T", T_1, "P", P_1, secondary_fluid)
            Q_pump = secondary_mdot*(h_2-h_1)/e_pump
    else:
        # State 1 is defined by the secondary minimum temperature and pressure
        P_1 = secondary_P_high/compression_ratio
        T_1 = secondary_min
        h_1 = PropsSI("H", "T", T_1, "P", P_1, secondary_fluid)
        s_1 = PropsSI("S", "T", T_1, "P", P_1, secondary_fluid)

        # Calculate the pump/compressor power
        P_2 = secondary_P_high
        if secondary_fluid == "Water" or secondary_fluid == "CarbonDioxide":
            # Approximate the fluid density with state 2
            T_2s = PropsSI("T", "S", s_1, "P", P_2, secondary_fluid)
            T_2 = T_1 + (T_2s-T_1)/e_pump
            h_2 = PropsSI("H", "T", T_2, "P", P_2, secondary_fluid)
            Q_pump = secondary_mdot*(h_2-h_1)/e_pump
        elif secondary_fluid == "Air":
            C_v = PropsSI("O", "T", T_1, "P", P_1, secondary_fluid)
            C_p = PropsSI("C", "T", T_1, "P", P_1, secondary_fluid)
            gamma = C_p/C_v
            T_2 = T_1*(compression_ratio**((gamma-1)/(gamma)))
            h_1 = PropsSI("H", "T", T_1, "P", P_1, secondary_fluid)
            Q_pump = secondary_mdot*(h_2-h_1)/e_pump

    # State 3 is defined by the HX hot temperature and operating pressure, and is immediately before entering the turbine
    P_3 = P_2
    T_3 = secondary_hot
    h_3 = PropsSI("H", "T", T_3, "P", P_3, secondary_fluid)
    s_3 = PropsSI("S", "T", T_3, "P", P_3, secondary_fluid)

    # State 4 is immediately after exiting the turbine
    P_4 = P_1
    if secondary_fluid == "Water" or secondary_fluid == "CarbonDioxide":
        h_4r = PropsSI("H", "S", s_3, "P", P_1, secondary_fluid)
        dH_3_4 = e_turbine*(h_3-h_4r)
        Q_turbine = secondary_mdot*dH_3_4*e_turbine
    elif secondary_fluid == "Air":
        C_v = PropsSI("O", "T", T_3, "P", P_3, secondary_fluid)
        C_p = PropsSI("C", "T", T_3, "P", P_3, secondary_fluid)
        gamma = C_p/C_v
        T_4 = T_3/(compression_ratio**((gamma-1)/gamma))
        h_4 = PropsSI("H", "T", T_4, "P", P_4, secondary_fluid)
        Q_turbine = secondary_mdot*(h_3-h_4)*e_turbine

    e_cycle = (Q_turbine-Q_pump)/Q_in

    results = {}
    results["Compressor Power (MW)"] = Q_pump/1000000
    results["Turbine Power (MW)"] = Q_turbine/1000000
    results["Cycle Efficiency"] = e_cycle

    return results
