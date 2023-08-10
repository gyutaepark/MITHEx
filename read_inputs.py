import pandas as pd
import warnings

def read_inputs():
    warnings.simplefilter(action='ignore', category=UserWarning)
    required_inputs = pd.read_excel("MITHEx_inputs.xlsx", sheet_name="Plant Description", index_col=0, header=None)
    optional_inputs = pd.read_excel("MITHEx_inputs.xlsx", sheet_name="HX Parameters", index_col=0, header=None)
    cycle_inputs = pd.read_excel("MITHEx_inputs.xlsx", sheet_name="Cycle Parameters", index_col=0, header=None)
    inputs = pd.concat([required_inputs, optional_inputs, cycle_inputs]).to_dict()[1]
    return inputs
