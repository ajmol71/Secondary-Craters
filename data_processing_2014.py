import pandas as pd
import time
import regex



# def parse_row(row):
    # crater_id = row[0:9]
    # lat_circ_img = row[11:17]
    # long_circ_img = row[19:26]
    # lat_circ_sd_img = row[28:33]
    # long_circ_sd_img = row[35:39]
    # lat_ell_img = row[41:46]
    # long_ell_img = row[48:55]
    # diam_circ_img = row[57:]
    # diam_circ_sd_img = row[]



with open('2014_data.txt', "r", encoding='utf-8', errors='ignore') as f:
    columns_14 = f.readline()



columns_list = columns_14.split()
num_cols = len(columns_list)
print(columns_list)

df_20 = pd.read_csv('2020_data.csv')
cols_list = list(df_20.columns)
data_2020 = [str(i) for i in df_20.iloc[0,:].tolist()]

# data_2014_df = pd.DataFrame(columns=columns_list)
#
# row_1 = data_2014.split()
#
# for j in range(len(columns_list)):
#     if j < len(cols_list):
#         print(j, columns_list[j], "  ", data_2014[j], "    ", cols_list[j], data_2020[j])
#     else:
#         print(j, columns_list[j], "  ", data_2014[j])

print()

# overlap_index_14 = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 25, 34, 35, 36, 37, 44]
# overlap_index_20 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23, 24, 29]
#
# for i in range(len(overlap_index_14)-1):
#     print(columns_list[overlap_index_14[i]], "  ", data_2014[overlap_index_14[i]], "  ", data_2020[overlap_index_20[i]])


def replace_data(row):
    row = row.replace("\t", "*")
    row = row.rstrip("\n")
    matches_obj = regex.finditer(r"\*{2,}", row)

    if row[-3:] == "***":
        flag = True
    else:
        flag = False

    og_locations = []
    replace_strs = []
    for match_obj in matches_obj:

        start_index = match_obj.start()
        end_index = match_obj.end()
        og_locations.append([start_index, end_index])

        match = match_obj.group()
        spaced = "\0".join(list(match))
        replace_strs.append(spaced)

    og_locations = og_locations[::-1]
    replace_strs = replace_strs[::-1]

    for j in range(len(og_locations)):
        og_ind = og_locations[j]
        row = row[:og_ind[0]] + replace_strs[j] + row[og_ind[1]:]

    if flag == True:
        row = row + "\0"

    new_data = row.split("*")
    return new_data


# print(matches)
# if more than one * in a row:
# every between ** would have  : num of missing data is one less than the number of *s
# Replace with spaces in order of appearance in string

full_table = pd.DataFrame(columns = columns_list)

time_start = time.time()

with open('2014_data.txt', "r", encoding='utf-8', errors='ignore') as f:
    columns_14 = f.readline()

    while True:
        new_line = f.readline()
        if not new_line.strip():
            break

        fixed_line = replace_data(new_line)
        full_table.loc[len(full_table)] = fixed_line

    full_table.to_csv("2014_data.csv")

run_time = time_start - time.time()
print("Time:", run_time)

# š
