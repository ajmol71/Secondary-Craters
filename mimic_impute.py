import numpy as np
import pandas as pd

data_20 = pd.read_csv("2020_data.csv")
secondaries = pd.read_csv("overlap_secondaries_crater_ids_adjusted.csv")["CRATER_ID_ADJ"].tolist()

data_20_craters = data_20["CRATER_ID"]

indices = []
data_20_imp = pd.DataFrame()
for col in data_20.columns:
    if col != "CRATER_ID":
        data_20_imp[col] = data_20[col]
        try:
            nanmean = np.nanmean(data_20_imp[col].tolist())
            data_20_imp[col + "_IMP_FLAG"] = np.where(data_20_imp[col].isna(), 1, 0)
            data_20_imp[col + "_IMP"] = np.where(data_20_imp[col + "_IMP_FLAG"] == 1, nanmean, data_20[col])
        except:
            continue

    else:
        data_20_imp[col] = data_20[col]

craters_list = []
for crater in data_20_craters:
    if crater in secondaries:
        craters_list.append(1)
    else:
        craters_list.append(0)


data_20_imp["SECONDARY"] = craters_list

# data_20_imp.to_csv("2020_data_imputed.csv")