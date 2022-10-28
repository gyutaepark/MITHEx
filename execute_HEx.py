import numpy as np
import pandas as pd
import read_inputs, compute_required_area

if __name__ == "__main__":
    inputs = read_inputs.read_inputs()
    results = compute_required_area.compute_required_area(inputs)
    print('Required area using the LMTD method is {}.'.format(results["Heat transfer area LMTD method"]))
    print('Required area using the e-NTU method is {}.'.format(results["Heat transfer area e-NTU method"]))
    #pd_results = pd.DataFrame.from_dict(results)
    #pd_results.to_csv('results.csv')
    