import pandas as pd
import math

data_14 = pd.read_csv("2014_data.csv")
data_20 = pd.read_csv("2020_data.csv")


# Extract Secondaries Only
secondaries_all = data_14["SECONDARY"]
expl = secondaries_all.tolist()[1]

secondaries_true_indicies = secondaries_all.isna()
secondaries_true = secondaries_all[~secondaries_true_indicies]
true_ids = data_14["CRATER_ID"][~secondaries_true_indicies]

# print(secondaries_true)
# print(secondaries_true.value_counts(normalize=False))



# Check if same craters and order in each dataset
craters_20 = data_20["CRATER_ID"].tolist()
craters_14_df = pd.read_csv("true_secondaries_crater_ids.csv")
cpp = craters_14_df["CRATER_ID"].tolist()

craters_14 = []
for i in cpp:
    craters_14.append(i[:5]+"0"+i[5:])

new_craters_14_df = pd.DataFrame(secondaries_true.index, columns=["INDEX"])
new_craters_14_df["CRATER_ID_ADJ"] = craters_14

target_num = len(craters_14)
total = 0
overlap_secondary_craters = []
for i in craters_14:
    if i in craters_20:
        total += 1
        overlap_secondary_craters.append(i)

print("Expected: ", target_num, "  Actual: ", total)

OL_craters_14_df = pd.DataFrame()
OL_craters_14_df["CRATER_ID_ADJ"] = overlap_secondary_craters
OL_craters_14_df.to_csv("overlap_secondaries_crater_ids_adjusted.csv")


