import pandas as pd

def read_inputs():
    required_inputs = pd.read_excel("MITHEx_inputs.xlsx", sheet_name="Plant Description", index_col=0, header=None)
    optional_inputs = pd.read_excel("MITHEx_inputs.xlsx", sheet_name="Optional Parameters", index_col=0, header=None)
    inputs = pd.concat([required_inputs, optional_inputs]).to_dict()[1]
    return inputs
