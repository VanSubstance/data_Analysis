from bs4 import BeautifulSoup
import requests
import csv
import pandas as pd
from selenium import webdriver

path_r = 'phase3\\leagues_c_final.csv'
data = pd.read_csv(path_r, encoding='utf-8', header=None)
players = data[0]

'''
여기서 파일경로를 바꿔주세요
'''
path_w1 = 'phase2\\' + path_r.split('\\')[-1].split('.')[0] + '-BC.csv'
f1 = open(path_w1, 'w', newline='', encoding='utf-8')
wr1 = csv.writer(f1)
wr1.writerow(['name', 'age', 'height', 'position', 'foot', 'date', 'mv'])

'''
여기서 파일경로를 바꿔주세요
'''
path_w2 = 'phase2\\' + path_r.split('\\')[-1].split('.')[0] + '-D.csv'
f2 = open(path_w2, 'w', newline='', encoding='utf-8')
wr2 = csv.writer(f2)
wr2.writerow(['name', 'season', 'apearance', 'goal', 'assist', 'penelty'])

for player in players:
    player = player.replace(' ', '+').replace('-','+')
    link = 'https://www.transfermarkt.co.uk/schnellsuche/ergebnis/schnellsuche?query=' + player
    headers = {'User-Agent': 'Custom'}
    response = requests.get(link, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    if len(soup) == 0:
        continue
    if not soup.select('a.spielprofil_tooltip'):
        continue   
    a = soup.select('a.spielprofil_tooltip')[0]
    url = 'https://www.transfermarkt.co.uk' + a.get('href')
    name = a.get('href').split('/')[1]
    print(name) #돌아가는지 확인용
    driver = webdriver.Chrome('chromedriver')
    driver.get(url)
    html = driver.page_source
    # table B,C merge된 친구 
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.select('table.auflistung')[0]
    li = []
    for tr in table.select('tr'):
        li.append(tr.get_text().strip())
    for l in li:
        if 'Age' in l.split(':')[0]:
            age = l.split(':')[1].strip()
        elif 'Height' in l.split(':')[0]:
            height = l.split(':')[1].strip().replace(',','.').replace('\xa0','')
        elif 'Position' in l.split(':')[0]:
            position = l.split(':')[1].strip()
        elif 'Foot' in l.split(':')[0]:
            foot = l.split(':')[1].strip()
    # tranfer history
    if not soup.select('div.box.transferhistorie'):
        continue       
    history=soup.select('div.box.transferhistorie')[0]
    tbody = history.select('tbody')[0]
    for tr in tbody.select('tr.zeile-transfer'):
        history_li = []
        for td in tr.select('td'):
            history_li.append(td.get_text())
        date = history_li[1].replace(',','')
        mv = history_li[10]
        wr1.writerow([name, age, height, position, foot, date, mv])
    driver.close()
    
    
    # table D    
    url = url.replace('profil', 'leistungsdatendetails')
    driver = webdriver.Chrome('chromedriver')
    driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    if soup.select('table.items') == []:
        continue
    table = soup.select('table.items')[0]
    tbody = table.select('tbody')[0]
    for tr in tbody.select('tr'):
        li2 = []
        for td in tr.select('td'):
            li2.append(td.get_text())
        season = li2[0]
        apearance = li2[4]
        goal = li2[6]
        assist = li2[7]
        penelty = li2[8].replace('\xa0','')
        wr2.writerow([name, season, apearance, goal, assist, penelty])
    driver.close()
    
f1.close()
f2.close()
print(nameL + "is complete!")