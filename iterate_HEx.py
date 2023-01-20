import numpy as np
import read_inputs, compute_required_area

if __name__ == "__main__":
    cost_results = []
    diameter_list = []
    primary_Re = []
    secondary_Re = []
    inputs = read_inputs.read_inputs()
    inflation = 1.62
    cost_conversion = 132000 # $/m3
    for primary_fluid in ["Sodium", "Helium"]:
        if primary_fluid == "Sodium":
            diameters = np.linspace(0.0005, 0.005, 50)
            inputs["Primary Pressure (kPa)"] = 300
        else:
            diameters = np.linspace(0.001, 0.005, 50)
            inputs["Primary Pressure (kPa)"] = 3000
        for secondary_fluid in ["Water", "CarbonDioxide", "Air"]:
            print(primary_fluid, secondary_fluid)
            inputs["Primary Fluid"] = primary_fluid
            inputs["Secondary Fluid"] = secondary_fluid
            inputs["HX length lower bound (m)"] = 0.05
            inputs["HX length upper bound (m)"] = 0.5

            if secondary_fluid == "CarbonDioxide":
                inputs["Secondary Pressure (kPa)"] = 25000
            else:
                inputs["Secondary Pressure (kPa)"] = 6000

            spec_results = []
            spec_secondary = []
            spec_primary = []

            for diameter in diameters:
                print(diameter)
                inputs["Channel Diameter (m)"] = diameter
                if primary_fluid == "Sodium":
                    if secondary_fluid == "Water":
                        if diameter > 0.0006:
                            inputs["HX length upper bound (m)"] = 0.4
                        if diameter > 0.00165:
                            inputs["HX length upper bound (m)"] = 0.6
                        if diameter > 0.0024:
                            inputs["HX length upper bound (m)"] = 0.8
                        if diameter > 0.0031:
                            inputs["HX length upper bound (m)"] = 1.2
                    else:
                        inputs["HX length upper bound (m)"] = 20
                else:
                    if secondary_fluid == "Water":
                        if diameter > 0.002:
                            inputs["HX length upper bound (m)"] = 1.2
                    else:
                        inputs["HX length upper bound (m)"] = 10

                print(inputs["HX length lower bound (m)"], inputs["HX length upper bound (m)"])

                if inputs["Secondary Fluid"] == "Water":
                    area_results = compute_required_area.compute_required_area_SG(inputs, verbose=False)
                else:
                    area_results = compute_required_area.compute_required_area(inputs)
                
                # Compute required heat exchanger volume and cost
                channel_thickness = 3 * inputs["Plate thickness (m)"] + inputs["Channel Diameter (m)"]
                channel_width = inputs["Plate thickness (m)"] + inputs["Channel Diameter (m)"]
                channel_volume = channel_thickness * channel_width * area_results["Heat Exchanger Length (m)"]
                HX_volume = channel_volume * area_results["Number of Channels"]
                cost = HX_volume * 132000 * inflation
                spec_results.append(cost)
                try:
                    spec_primary.append(area_results["Primary Re"])
                    spec_secondary.append(area_results["Secondary Re"])
                except:
                    spec_primary.append(area_results["Primary inlet Re"])
                    spec_secondary.append(area_results["Secondary inlet Re"])
            
            cost_results.append(spec_results)
            primary_Re.append(spec_primary)
            secondary_Re.append(spec_secondary)
            diameter_list.append(diameters / 1000)

    print(cost_results)
    print(diameter_list)
    