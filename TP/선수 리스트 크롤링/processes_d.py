import pandas as pd

data = pd.read_csv('phase2\\leagues_bc.csv')

# date 
data['date'] = pd.to_datetime(data['date'])

# heigth . 떼고 m떼서 cm 단위의 int
data['height'] = data['height'].str.replace('.','')
data['height'] = data['height'].str.replace('m','')
zero = pd.Series(['0']*data.shape[0])
data['height'] = data['height'].str.cat(zero)
data['height'] = data['height'].str.slice(0,3).apply(pd.to_numeric)

# age 계산
today = pd.DataFrame([pd.datetime(2020,11,10)]*data.shape[0])
data['age'] = data['age'] - (today[0] - data['date']).astype('timedelta64[Y]')

data.to_csv("leagues_c.csv", index = False)

#%% d file
data_d = pd.read_csv(r'C:\Users\yuwoo\Desktop\3-2\6 Biz Analysis\팀플\phase2\leagues_d.csv', encoding='utf-8')

#%% goal keeper drop
goal_keeper = data['name'][data['position']=='Goalkeeper']
goal_keeper = goal_keeper.drop_duplicates()
ind = []
for gk in goal_keeper:
    ind.extend(data_d[data_d['name']==gk].index)
data_d = data_d.drop(index=ind)

drop = data_d[data_d['assist'].str.len() >3].index
data_d = data_d.drop(index=drop)

#%% penelty divide
penelty = data_d['penelty'].str.split('/', expand=True)
data_d['yellow'] = penelty[0]
data_d['yellow2'] = penelty[1]
data_d['red'] = penelty[2]
data_d = data_d.drop('penelty', axis=1)

change = data_d[data_d['yellow'].str.len() > 2].index
data_d['yellow'][change] = '0'

#%% make to int
int_columns = ['apearance', 'goal', 'assist', 'yellow','yellow2', 'red']
for c in int_columns:
    data_d[c] = data_d[c].fillna('0')
    data_d[c] = data_d[c].str.replace('-','0')
    data_d[c] = data_d[c].astype('int')

#%% season
wrong_index = data_d[data_d['season'].str.len() > 5].index
wrong_split = data_d['season'][wrong_index].str.split(' ', expand=True)
wrong_split =  wrong_split[0].str.slice(0,2) +'/'+ wrong_split[1].str.slice(0,2)
data_d['season'][wrong_index] = wrong_split

no_slash = data_d[data_d['season'].str.len() < 5].index
data_d['season'][no_slash] = data_d['season'][no_slash].str.slice(2,4)
yes_slash = data_d[data_d['season'].str.len() >3].index
data_d['season'][yes_slash] = data_d['season'][yes_slash].str.slice(0,2)
data_d['season']
data_d = data_d.drop(index=data_d[data_d['season']=='48'].index)
data_d = data_d.drop(index=data_d[data_d['season']=='52'].index)

players = data_d['name'].drop_duplicates()
data_d_final = pd.DataFrame(columns = ['season', 'apearance', 'goal', 'assist', 'yellow', 'yellow2', 'red','name'])
for p in players:
    player_table = data_d[data_d['name'] == p]
    player_table = player_table.set_index('season')
    player_table = player_table.sum(level='season')
    player_table = player_table.reset_index()
    player_table['name'] = [p]*player_table.shape[0]
    data_d_final = data_d_final.append(player_table, ignore_index=True)