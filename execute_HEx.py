from units import Q_
import numpy as np
import pandas as pd
import datetime
import read_inputs, compute_required_area, compute_cycle_efficiency

if __name__ == "__main__":
    inputs = read_inputs.read_inputs()
    if inputs["Secondary Fluid"] == "Water":
        area_results = compute_required_area.compute_required_area_SG(inputs)
        channel_thickness = 3 * inputs["Plate thickness (m)"] + inputs["Channel Diameter (m)"]
        channel_width = inputs["Plate thickness (m)"] + inputs["Channel Diameter (m)"]
        channel_volume = channel_thickness * channel_width * area_results["Heat Exchanger Length"]
        HX_volume = channel_volume * area_results["Number of Channels"]
        cost = HX_volume * 132000 * 1.62
        cost_results = {
            "HX Volume (m**3)": HX_volume,
            "HX cost": cost
        }
    else:
        area_results = compute_required_area.compute_required_area(inputs)
        # Compute required heat exchanger volume and cost
        thickness = Q_(3 * inputs["Plate thickness (m)"] + inputs["Channel Diameter (m)"], 'm')
        V_LMTD = area_results["Heat transfer area LMTD method"] * thickness
        cost_LMTD = V_LMTD * Q_(132000, '1 / m**3') * 1.62
        V_eNTU = area_results["Heat transfer area e-NTU method"] * thickness
        cost_eNTU = V_LMTD * Q_(132000, '1 / m**3') * 1.62
        cost_results = {
            "HX Volume LMTD (m**3)": V_LMTD,
            "HX Volume e-NTU (m**3)": V_eNTU,
            "HX Cost LMTD ($)": cost_LMTD.m,
            "HX Cost e-NTU ($)": cost_eNTU.m
        }
    inputs["Primary Mass Flow Rate (kg/s)"] = area_results["Primary Mass Flow Rate (kg/s)"]
    inputs["Secondary Mass Flow Rate (kg/s)"] = area_results["Secondary Mass Flow Rate (kg/s)"]

    cycle_results = compute_cycle_efficiency.compute_cycle_efficiency(inputs)

    results = area_results | cost_results | cycle_results

    pd_results = pd.Series(results)
    print(pd_results)
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pd_results.to_csv('results/' + t + '.csv')
    