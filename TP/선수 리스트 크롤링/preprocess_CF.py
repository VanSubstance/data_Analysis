#%% MV 바꾸기 -> 단위 파운드로 다 맞추기

import pandas as pd

dropper = []

#%% Fill up all blank mv
data = pd.read_csv("phase2\\" + "leagues_bc" + ".csv", error_bad_lines=False)
dataBC = pd.read_csv("phase2\\" + "leagues_bc" + ".csv", error_bad_lines=False)
dataBC = dataBC.drop_duplicates()
for player in dataBC['name'].unique():
    indexes = dataBC[dataBC['name'] == player].index
    part = []
    parts = []
    for seq in range(len(indexes) - 1):
        if seq == len(indexes) - 2:
            part.append(indexes[seq + 1])
            parts.append(part)
            part = []
        if indexes[seq] == indexes[seq + 1]:
            part.append(indexes[seq])
        else:
            parts.append(part)
            part = []
            part.append(indexes[seq])
    for part in parts:
        for seq in part:
            if not (dataBC[dataBC['name'] == player]['mv'][seq] == "-"):
                limit = dataBC[dataBC['name'] == player]['mv'][seq]
    if not limit:
        dropper.append(player)
        index = dataBC[dataBC['name'] == player].index
        dataBC = dataBC.drop(index)
    else:
        dataBC.loc[dataBC['name'] == player, 'mv'] = dataBC[dataBC['name'] == player]['mv'].str.replace("-", limit)
        print(player + " done!")

print("Filling up the blanks is done!")


#%% Change mv into number

def mv(x):
    print(x)
    if x[-1] == "m":
        x = x[:-1]
        x = float(x) * 1000000
    else:
        x = x[:-3]
        x = float(x) * 1000
    return x

dataBC['mv'] = dataBC['mv'].str.replace("£", "")
dataBC['mv'] = dataBC['mv'].apply(mv)

#%% Changing year into digit 
def year_digit(x):
    return x[2:4]
dataBC['date'] = dataBC['date'].apply(year_digit)

#%% F table from BC
dataF = dataBC[['name', 'foot', 'height', 'position']]
dataBC.to_csv("phase3\\" + "leagues_c" + ".csv", index = True, columns = ["name", "age", "date", "mv", "potential"])
dataF.to_csv("phase3\\" + "leagues_f" + ".csv", index = True)
