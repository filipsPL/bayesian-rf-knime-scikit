import pandas as pd

features = input_table.iloc[:,:-1]		# data - all but the last column
target = input_table.iloc[:,-1]		# class - the last column


# Predict using existing model + new data
predictions = pd.DataFrame(input_model.predict_proba(features))

# join with the input_table
joined_table = pd.concat([input_table.reset_index(),predictions.reset_index(drop=True)], axis=1)

# rename headers to be consistent with WEKA and KNIME (to some degree, of course)
output_table = joined_table.rename(columns={0: "P (class=0)", 1: "P (class=1)"}).set_index('index')
