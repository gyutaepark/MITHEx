import numpy as np
import pandas as pd
import datetime
import read_inputs, compute_required_area

if __name__ == "__main__":
    inputs = read_inputs.read_inputs()
    results = compute_required_area.compute_required_area(inputs)
    pd_results = pd.Series(results)
    print(pd_results)
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pd_results.to_csv('results/' + t + '.csv')
    