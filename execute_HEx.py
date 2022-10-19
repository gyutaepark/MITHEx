import numpy as np
import read_inputs, compute_required_area

if __name__ == "__main__":
    primary_fluids = ["Helium", "Sodium", "Organic", "Heat Pipe"]
    secondary_fluids = ["Air", "SCO2", "Steam"]
    pcs = {
        "Air": "Open Air Brayton Cycle",
        "SCO2": "Supercritical CO2 Cycle",
        "Steam": "Steam Rankine Cycle"
    }
    inputs = read_inputs.read_inputs(primary_fluids, secondary_fluids, pcs)
    hx_area = compute_required_area.compute_required_area(inputs)
    