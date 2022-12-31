from units import Q_
import numpy as np
import pandas as pd
import datetime
import read_inputs, compute_required_area, compute_cycle_efficiency

if __name__ == "__main__":
    inputs = read_inputs.read_inputs()
    inflation = 1.62
    cost_conversion = 132000 # $/m3
    if inputs["Secondary Fluid"] == "Water":
        area_results = compute_required_area.compute_required_area_SG(inputs)
    else:
        area_results = compute_required_area.compute_required_area(inputs)
    
    # Compute required heat exchanger volume and cost
    channel_thickness = 3 * inputs["Plate thickness (m)"] + inputs["Channel Diameter (m)"]
    channel_width = inputs["Plate thickness (m)"] + inputs["Channel Diameter (m)"]
    channel_volume = channel_thickness * channel_width * area_results["Heat Exchanger Length (m)"]
    HX_volume = channel_volume * area_results["Number of Channels"]
    cost = HX_volume * 132000 * inflation
    cost_results = {
        "HX Volume (m**3)": HX_volume,
        "HX cost": cost
    }

    inputs["Primary Mass Flow Rate (kg/s)"] = area_results["Primary Mass Flow Rate (kg/s)"]
    inputs["Secondary Mass Flow Rate (kg/s)"] = area_results["Secondary Mass Flow Rate (kg/s)"]

    cycle_results = compute_cycle_efficiency.compute_cycle_efficiency(inputs)

    results = area_results | cost_results | cycle_results

    pd_results = pd.Series(results)
    print(pd_results)
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pd_results.to_csv('results/' + t + '.csv')
    