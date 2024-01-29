from units import Q_
import numpy as np
import sdf
import scipy.optimize
from CoolProp.CoolProp import PropsSI
from th_functions import fluid_properties, saturated_liquid_properties, saturated_vapor_properties, compute_Nu_Dittus_Boelter, compute_Nu_Gnielinski, \
    compute_Nu_Seban_Shimazaki, compute_Nu_Chen, compute_k_SS, compute_k_SS304, compute_k_SS316, compute_rho_SS, compute_NTU_counterflow, \
    compute_friction_factor, compute_equilibrium_quality, compute_flow_quality, compute_htc_Dittus_Boelter, compute_two_phase_friction_multiplier, \
    compute_pressure_drop

def compute_required_area(inputs):
    # initialize thermal inputs
    # yapf:disable
    Q              = Q_(inputs["Thermal Power (MW)"], 'MW').m_as('W')
    primary_fluid  = inputs["Primary Fluid"]
    primary_hot    = Q_(inputs["Primary Hot Temperature (C)"], 'degC').m_as('K')
    primary_cold   = Q_(inputs["Primary Cold Temperature (C)"], 'degC').m_as('K')
    primary_mdot   = inputs["Primary Mass Flow Rate (kg/s)"]
    primary_P      = Q_(inputs["Primary Pressure (kPa)"], 'kPa').m_as('Pa')
    primary_deltaP = Q_(inputs["Primary Pressure Drop (kPa)"], 'kPa').m_as('Pa')

    intermediate_fluid           = "Sodium"
    intermediate_hot             = Q_(inputs["Intermediate Hot Temperature (C)"], 'degC').m_as('K')
    intermediate_cold            = Q_(inputs["Intermediate Cold Temperature (C)"], 'degC').m_as('K')
    intermediate_P               = primary_P
    intermediate_deltaP          = primary_deltaP
    intermediate_avg_temperature = (intermediate_hot+intermediate_cold)/2
    intermediate_cp, intermediate_k, intermediate_rho, intermediate_mu = fluid_properties(intermediate_fluid, intermediate_avg_temperature, intermediate_P)
    intermediate_mdot            = fill_out_parameters(Q, intermediate_cp, T_hot=intermediate_hot, T_cold=intermediate_cold)

    secondary_fluid  = inputs["Secondary Fluid"]
    secondary_hot    = Q_(inputs["Secondary Hot Temperature (C)"], 'degC').m_as('K')
    secondary_cold   = Q_(inputs["Secondary Cold Temperature (C)"], 'degC').m_as('K')
    secondary_mdot   = inputs["Secondary Mass Flow Rate (kg/s)"]
    secondary_P      = Q_(inputs["Secondary Pressure (kPa)"], 'kPa').m_as('Pa')
    secondary_deltaP = Q_(inputs["Secondary Pressure Drop (kPa)"], 'kPa').m_as('Pa')
    # fill out missing thermal parameters
    if np.isnan(primary_mdot):
        primary_avg_temperature = (primary_hot+primary_cold)/2
        primary_cp, primary_k, primary_rho, primary_mu = fluid_properties(primary_fluid, primary_avg_temperature, primary_P)
        primary_mdot = fill_out_parameters(Q, primary_cp, T_hot=primary_hot, T_cold=primary_cold)
    elif np.isnan(primary_hot):
        primary_cp, primary_k, primary_rho, primary_mu = fluid_properties(primary_fluid, primary_cold, primary_P)
        primary_hot = fill_out_parameters(Q, primary_cp, T_cold=primary_cold, m_dot=primary_mdot)
    elif np.isnan(primary_cold):
        primary_cp, primary_k, primary_rho, primary_mu = fluid_properties(primary_fluid, primary_hot, primary_P)
        primary_cold = fill_out_parameters(Q, primary_cp, T_hot=primary_hot, m_dot=primary_mdot)

    # fill out missing thermal parameters
    if np.isnan(secondary_mdot):
        secondary_avg_temperature = (secondary_hot+secondary_cold)/2
        secondary_cp, secondary_k, secondary_rho, secondary_mu = fluid_properties(secondary_fluid, secondary_avg_temperature, secondary_P)
        secondary_mdot = fill_out_parameters(Q, secondary_cp, T_hot=secondary_hot, T_cold=secondary_cold)
    elif np.isnan(secondary_hot):
        secondary_cp, secondary_k, secondary_rho, secondary_mu = fluid_properties(secondary_fluid, secondary_cold, secondary_P)
        secondary_hot = fill_out_parameters(Q, secondary_cp, T_cold=secondary_cold, m_dot=secondary_mdot)
    elif np.isnan(secondary_cold):
        secondary_cp, secondary_k, secondary_rho, secondary_mu = fluid_properties(secondary_fluid, secondary_hot, secondary_P)
        secondary_cold = fill_out_parameters(Q, secondary_cp, T_hot=secondary_hot, m_dot=secondary_mdot)
    # yapf:enable
    # Initialize geometry
    plate_thickness = np.array([
        inputs["HX1 Plate thickness (m)"],
        inputs["HX2 Plate thickness (m)"],
    ])
    plate_material = np.array([
        inputs["HX1 Plate material"],
        inputs["HX2 Plate material"],
    ])
    D = inputs["HX Channel Diameter (m)"]
    flow_area = np.pi*(D/2)**2/2
    perimeter = D + np.pi*D/2
    D_h = 4*flow_area/perimeter

    # Compute plate thermal conductivity
    T1_avg = (primary_hot+primary_cold+intermediate_hot+intermediate_cold)/4
    T2_avg = (intermediate_hot+intermediate_cold+secondary_hot+secondary_cold)/4

    plate_k1 = compute_k_SS(T1_avg, plate_material[0])
    plate_k2 = compute_k_SS(T2_avg, plate_material[1])

    def determine_HX_length(L, searching=True):
        # Determine flow velocities by finding the limiting pressure drop
        # Both fluids should have an equal fraction of mass flow rate in each channel
        L1, L2 = L
        primary_velocity = scipy.optimize.brentq(
            _compute_velocity,
            1e-3,
            1e6,
            args=(primary_rho, D_h, primary_mu, L1, primary_deltaP))
        primary_mdot_channel = primary_rho*primary_velocity*flow_area

        intermediate_velocity = scipy.optimize.brentq(
            _compute_velocity,
            1e-3,
            1e6,
            args=(intermediate_rho,
                  D_h,
                  intermediate_mu,
                  L2,
                  intermediate_deltaP))
        intermediate_mdot_channel = intermediate_rho*intermediate_velocity*flow_area

        secondary_velocity = scipy.optimize.brentq(
            _compute_velocity,
            1e-3,
            1e6,
            args=(secondary_rho, D_h, secondary_mu, L2, secondary_deltaP))
        secondary_mdot_channel = secondary_rho*secondary_velocity*flow_area

        primary_mdot_ratio = primary_mdot_channel/primary_mdot
        intermediate_mdot_ratio = intermediate_mdot_channel/intermediate_mdot
        secondary_mdot_ratio = secondary_mdot_channel/secondary_mdot

        ratio_dict = {
            'primary_mdot_ratio': primary_mdot_ratio,
            'intermediate_mdot_ratio': intermediate_mdot_ratio,
            'secondary_mdot_ratio': secondary_mdot_ratio
        }
        min_ratio = min(ratio_dict, key=ratio_dict.get)
        if min_ratio == 'primary_mdot_ratio':
            intermediate_mdot_channel = intermediate_mdot*primary_mdot_ratio
            intermediate_velocity = intermediate_mdot_channel/(
                flow_area*intermediate_rho)
            secondary_mdot_channel = secondary_mdot*primary_mdot_ratio
            secondary_velocity = secondary_mdot_channel/(
                flow_area*secondary_rho)
        elif min_ratio == 'intermediate_mdot_ratio':
            primary_mdot_channel = primary_mdot*intermediate_mdot_ratio
            primary_velocity = primary_mdot_channel/(flow_area*primary_rho)
            secondary_mdot_channel = secondary_mdot*intermediate_mdot_ratio
            secondary_velocity = secondary_mdot_channel/(
                flow_area*secondary_rho)
        else:
            primary_mdot_channel = primary_mdot*secondary_mdot_ratio
            primary_velocity = primary_mdot_channel/(flow_area*primary_rho)
            intermediate_mdot_channel = intermediate_mdot*secondary_mdot_ratio
            intermediate_velocity = intermediate_mdot_channel/(
                flow_area*intermediate_rho)

        # Calculate heat transfer coefficients
        primary_Re = primary_rho*primary_velocity*D_h/primary_mu
        primary_Pr = primary_mu*primary_cp/primary_k
        if primary_fluid == "Sodium":
            primary_Pe = primary_Re*primary_Pr
            primary_Nu = compute_Nu_Seban_Shimazaki(primary_Pe)
        else:
            primary_f = compute_friction_factor(primary_Re)
            if primary_Re > 2300:
                primary_Nu = compute_Nu_Gnielinski(primary_Re,
                                                   primary_Pr,
                                                   primary_f)
            else:
                primary_Nu = 4.036
        primary_h = primary_Nu*primary_k/D_h

        intermediate_Re = intermediate_rho*intermediate_velocity*D_h/intermediate_mu
        intermediate_Pr = intermediate_mu*intermediate_cp/intermediate_k
        if intermediate_fluid == "Sodium":
            intermediate_Pe = intermediate_Re*intermediate_Pr
            intermediate_Nu = compute_Nu_Seban_Shimazaki(intermediate_Pe)
        else:
            intermediate_f = compute_friction_factor(intermediate_Re)
            if intermediate_Re > 2300:
                intermediate_Nu = compute_Nu_Gnielinski(intermediate_Re,
                                                        intermediate_Pr,
                                                        intermediate_f)
            else:
                intermediate_Nu = 4.036
        intermediate_h = intermediate_Nu*intermediate_k/D_h

        secondary_Re = secondary_rho*secondary_velocity*D_h/secondary_mu
        secondary_Pr = secondary_mu*secondary_cp/secondary_k
        secondary_f = compute_friction_factor(secondary_Re)
        if secondary_Re > 2300:
            secondary_Nu = compute_Nu_Gnielinski(secondary_Re,
                                                 secondary_Pr,
                                                 secondary_f)
        else:
            secondary_Nu = 4.036
        secondary_h = secondary_Nu*secondary_k/D_h

        # Calculate overall heat transfer coefficient
        U1 = (primary_h**-1 + plate_thickness[0]*plate_k1**-1
              + intermediate_h**-1)**-1
        U2 = (intermediate_h**-1 + plate_thickness[1]*plate_k2**-1
              + secondary_h**-1)**-1

        # LMTD METHOD
        # Calculate LMTD
        deltaT_1h = primary_hot - intermediate_hot
        deltaT_1c = primary_cold - intermediate_cold
        LMTD1 = (deltaT_1h-deltaT_1c)/np.log(deltaT_1h/deltaT_1c)

        deltaT_2h = intermediate_hot - secondary_hot
        deltaT_2c = intermediate_cold - secondary_cold
        LMTD2 = (deltaT_2h-deltaT_2c)/np.log(deltaT_2h/deltaT_2c)

        # Calculate required area for a single channel
        Q_single_channel1 = primary_mdot_channel*primary_cp*(
            primary_hot-primary_cold)
        A_required1 = Q_single_channel1/(U1*LMTD1)
        L_calculated1 = A_required1/D

        Q_single_channel2 = intermediate_mdot_channel*intermediate_cp*(
            intermediate_hot-intermediate_cold)
        A_required2 = Q_single_channel2/(U2*LMTD2)
        L_calculated2 = A_required2/D

        if not searching:
            return Q_single_channel

        return [(L1 - L_calculated1), (L2 - L_calculated2)]

    # Initialize bounds of heat exchanger length solver
    L_lower_bound = inputs["HX length lower bound (m)"]
    L_upper_bound = inputs["HX length upper bound (m)"]
    res = scipy.optimize.root(determine_HX_length,
                              [L_lower_bound, L_upper_bound])
    HX_Ls = res.x

    # Determine flow velocities by finding the limiting pressure drop
    # Both fluids should have an equal fraction of mass flow rate in each channel
    primary_velocity = scipy.optimize.brentq(
        _compute_velocity,
        1e-3,
        1e6,
        args=(primary_rho, D_h, primary_mu, HX_Ls[0], primary_deltaP))
    primary_mdot_channel = primary_rho*primary_velocity*flow_area

    intermediate_velocity = scipy.optimize.brentq(
        _compute_velocity,
        1e-3,
        1e6,
        args=(intermediate_rho,
              D_h,
              intermediate_mu,
              HX_Ls[1],
              intermediate_deltaP))
    intermediate_mdot_channel = intermediate_rho*intermediate_velocity*flow_area

    secondary_velocity = scipy.optimize.brentq(
        _compute_velocity,
        1e-3,
        1e6,
        args=(secondary_rho, D_h, secondary_mu, HX_Ls[1], secondary_deltaP))
    secondary_mdot_channel = secondary_rho*secondary_velocity*flow_area

    primary_mdot_ratio = primary_mdot_channel/primary_mdot
    intermediate_mdot_ratio = intermediate_mdot_channel/intermediate_mdot
    secondary_mdot_ratio = secondary_mdot_channel/secondary_mdot

    ratio_dict = {
        'primary_mdot_ratio': primary_mdot_ratio,
        'intermediate_mdot_ratio': intermediate_mdot_ratio,
        'secondary_mdot_ratio': secondary_mdot_ratio
    }
    min_ratio = min(ratio_dict, key=ratio_dict.get)
    if min_ratio == 'primary_mdot_ratio':
        intermediate_mdot_channel = intermediate_mdot*primary_mdot_ratio
        intermediate_velocity = intermediate_mdot_channel/(
            flow_area*intermediate_rho)
        secondary_mdot_channel = secondary_mdot*primary_mdot_ratio
        secondary_velocity = secondary_mdot_channel/(flow_area*secondary_rho)
    elif min_ratio == 'intermediate_mdot_ratio':
        primary_mdot_channel = primary_mdot*intermediate_mdot_ratio
        primary_velocity = primary_mdot_channel/(flow_area*primary_rho)
        secondary_mdot_channel = secondary_mdot*intermediate_mdot_ratio
        secondary_velocity = secondary_mdot_channel/(flow_area*secondary_rho)
    else:
        primary_mdot_channel = primary_mdot*secondary_mdot_ratio
        primary_velocity = primary_mdot_channel/(flow_area*primary_rho)
        intermediate_mdot_channel = intermediate_mdot*secondary_mdot_ratio
        intermediate_velocity = intermediate_mdot_channel/(
            flow_area*intermediate_rho)

    # Calculate heat transfer coefficients
    primary_Re = primary_rho*primary_velocity*D_h/primary_mu
    primary_Pr = primary_mu*primary_cp/primary_k
    if primary_fluid == "Sodium":
        primary_Pe = primary_Re*primary_Pr
        primary_Nu = compute_Nu_Seban_Shimazaki(primary_Pe)
    else:
        primary_f = compute_friction_factor(primary_Re)
        if primary_Re > 2300:
            primary_Nu = compute_Nu_Gnielinski(primary_Re,
                                               primary_Pr,
                                               primary_f)
        else:
            primary_Nu = 4.036
    primary_h = primary_Nu*primary_k/D_h

    intermediate_Re = intermediate_rho*intermediate_velocity*D_h/intermediate_mu
    intermediate_Pr = intermediate_mu*intermediate_cp/intermediate_k
    if intermediate_fluid == "Sodium":
        intermediate_Pe = intermediate_Re*intermediate_Pr
        intermediate_Nu = compute_Nu_Seban_Shimazaki(intermediate_Pe)
    else:
        intermediate_f = compute_friction_factor(intermediate_Re)
        if intermediate_Re > 2300:
            intermediate_Nu = compute_Nu_Gnielinski(intermediate_Re,
                                                    intermediate_Pr,
                                                    intermediate_f)
        else:
            intermediate_Nu = 4.036
    intermediate_h = intermediate_Nu*intermediate_k/D_h

    secondary_Re = secondary_rho*secondary_velocity*D_h/secondary_mu
    secondary_Pr = secondary_mu*secondary_cp/secondary_k
    secondary_f = compute_friction_factor(secondary_Re)
    if secondary_Re > 2300:
        secondary_Nu = compute_Nu_Gnielinski(secondary_Re,
                                             secondary_Pr,
                                             secondary_f)
    else:
        secondary_Nu = 4.036
    secondary_h = secondary_Nu*secondary_k/D_h

    # Calculate overall heat transfer coefficient
    U1 = (primary_h**-1 + plate_thickness[0]*plate_k1**-1
          + intermediate_h**-1)**-1
    U2 = (intermediate_h**-1 + plate_thickness[1]*plate_k2**-1
          + secondary_h**-1)**-1

    # LMTD METHOD
    # Calculate LMTD
    deltaT_1h = primary_hot - intermediate_hot
    deltaT_1c = primary_cold - intermediate_cold
    LMTD1 = (deltaT_1h-deltaT_1c)/np.log(deltaT_1h/deltaT_1c)

    deltaT_2h = intermediate_hot - secondary_hot
    deltaT_2c = intermediate_cold - secondary_cold
    LMTD2 = (deltaT_2h-deltaT_2c)/np.log(deltaT_2h/deltaT_2c)

    # Calculate heat exchange area for a single channel
    A1 = D*HX_Ls[0]
    A2 = D*HX_Ls[1]

    Q_single_channel1 = U1*A1*LMTD1
    Q_single_channel2 = U2*A2*LMTD2

    num_channels1 = Q/Q_single_channel1
    num_channels2 = Q/Q_single_channel2

    # Calculate pressure drop for each fluid
    primary_dP = compute_pressure_drop(
        primary_velocity,
        primary_rho,
        D_h,
        primary_mu,
        HX_Ls[0],
    )
    intermediate_dP1 = compute_pressure_drop(
        intermediate_velocity,
        intermediate_rho,
        D_h,
        intermediate_mu,
        HX_Ls[0],
    )
    intermediate_dP2 = compute_pressure_drop(
        intermediate_velocity,
        intermediate_rho,
        D_h,
        intermediate_mu,
        HX_Ls[1],
    )
    secondary_dP = compute_pressure_drop(
        secondary_velocity,
        secondary_rho,
        D_h,
        secondary_mu,
        HX_Ls[1],
    )
    primary_mass_flux = primary_mdot_channel/flow_area
    intermediate_mass_flux = intermediate_mdot_channel/flow_area
    secondary_mass_flux = secondary_mdot_channel/flow_area
    # Channel refers to two half-circle channels combined
    channel_thicknesses = 3*plate_thickness + D
    channel_widths = plate_thickness + D
    channel_volumes = channel_thicknesses*channel_widths*HX_Ls
    channel_volumes_tot = np.pi*(D/2)**2*HX_Ls*np.array(
        [num_channels1, num_channels2])
    HX_volumes = channel_volumes*np.array([num_channels1, num_channels2])
    HX_volumes_wo_channels = HX_volumes - channel_volumes_tot
    HX_masses = (
        np.array([
            compute_rho_SS(plate_material[0]),
            compute_rho_SS(plate_material[1])
        ])*HX_volumes_wo_channels)

    # Store results to be displayed
    # yapf:disable
    results = {}
    results["Primary Re"]                                 = primary_Re
    results["Intermediate Re"]                            = intermediate_Re
    results["Secondary Re"]                               = secondary_Re
    results["Primary Nu"]                                 = primary_Nu
    results["Intermediate Nu"]                            = intermediate_Nu
    results["Secondary Nu"]                               = secondary_Nu
    results["Primary htc (W/m2)"]                         = primary_h
    results["Intermediate htc (W/m2)"]                    = intermediate_h
    results["Secondary htc (W/m2)"]                       = secondary_h
    results["Primary Velocity (m/s)"]                     = primary_velocity
    results["Intermediate Velocity (m/s)"]                = intermediate_velocity
    results["Secondary Velocity"]                         = secondary_velocity
    results["Primary Channel Mass Flow Rate (kg/s)"]      = primary_mdot_channel
    results["Intermediate Channel Mass Flow Rate (kg/s)"] = intermediate_mdot_channel
    results["Secondary Channel Mass Flow Rate (kg/s)"]    = secondary_mdot_channel
    results["Primary Mass Flow Rate (kg/s)"]              = primary_mdot
    results["Intermediate Mass Flow Rate (kg/s)"]         = intermediate_mdot
    results["Intermediate Volumetric Flow Rate (m3/s)"]   = intermediate_mdot/intermediate_rho
    results["Secondary Mass Flow Rate (kg/s)"]            = secondary_mdot
    results["Primary Mass Flux (kg/m2/s)"]                = primary_mass_flux
    results["Intermediate Mass Flux (kg/m2/s)"]           = intermediate_mass_flux
    results["Secondary Mass Flux (kg/m2/s)"]              = secondary_mass_flux
    results["Primary HX Pressure Drop (Pa)"]              = primary_dP
    results["Intermediate HX1 Pressure Drop (Pa)"]        = intermediate_dP1
    results["Intermediate HX2 Pressure Drop (Pa)"]        = intermediate_dP2
    results["Secondary HX Pressure Drop (Pa)"]            = secondary_dP
    results["Overall HTC 1 (W/m2/s)"]                     = U1
    results["Overall HTC 2 (W/m2/s)"]                     = U2
    results["HX1 Number of Channels"]                     = num_channels1
    results["HX2 Number of Channels"]                     = num_channels2
    results["HX1 Length (m)"]                             = HX_Ls[0]
    results["HX2 Length (m)"]                             = HX_Ls[1]
    results["HX1 Coolant Volume (m3)"]                    = channel_volumes_tot[0]
    results["HX2 Coolant Volume (m3)"]                    = channel_volumes_tot[1]
    results["HX1 Volume (m3)"]                            = HX_volumes[0]
    results["HX2 Volume (m3)"]                            = HX_volumes[1]
    results["HX1 Mass (kg)"]                              = HX_masses[0]
    results["HX2 Mass (kg)"]                              = HX_masses[1]
    # yapf:enable
    return results

def _compute_velocity(v, rho, D, mu, L, deltaP, rho_1=None, rho_2=None):
    Re = rho*v*D/mu
    f = compute_friction_factor(Re)
    dP = 0.5*f*rho*v**2*L/D

    if rho_1 is not None:
        v_1 = rho*v/rho_1
        v_2 = rho*v/rho_2
        dP_acc = rho_2*v_2**2 - rho*v**2
        dP += dP_acc

    return (deltaP - dP)

def fill_out_parameters(Q, c_p, T_hot=None, T_cold=None, m_dot=None):
    """
    This function takes the rated power and specific heat of a fluid,
    along with a required two input parameters and uses those to calculate
    the missing parameter.
    """
    if T_hot == None:
        return Q/(m_dot*c_p) + T_cold
    if T_cold == None:
        return T_hot - Q/(m_dot*c_p)
    if m_dot == None:
        return Q/(c_p*(T_hot-T_cold))
    raise KeyError("Too many inputs. Two optional inputs are required.")
