# importing python package
import pandas as pd

filename = 'data.csv'

from sklearn.model_selection import train_test_split

import csv

train,test = train_test_split(pd.read_csv(filename), test_size=0.30, random_state=0)

train.to_csv('train.csv',index=False)
test.to_csv('test.csv',index=False)

datasets = ['train.csv', 'test.csv']

for i in datasets:
    with open(i,newline='', encoding = "ISO-8859-1") as f:
        r = csv.reader(f)
        data = [line for line in r]
    with open(i,'w',newline='', encoding = "ISO-8859-1") as f:
        w = csv.writer(f)
        w.writerow(['label', 'text'])
        w.writerows(data)
