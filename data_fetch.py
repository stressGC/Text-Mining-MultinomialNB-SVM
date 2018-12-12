"""
Script used to fetch the base dataset into something smaller
keeps only the features we want to work with
converts JSON to CSV for easier use with Python/pandas
"""

import pandas as pd
import time
import ast

# lets read the entire dataset
with open('data/phones.json', 'rb') as f:
    data = f.readlines()
    
# remove the trailing "\n" from each element
data = list(map(lambda x: x.rstrip(), data))
print('number reviews:', len(data))

# for now, let's restrict to the first 50k observations
data = data[:50000]

# convert list to a dataframe
t1 = time.time()
df = pd.DataFrame()
count = 0

for r in data:
    r = r.decode('utf-8')
    r = ast.literal_eval(r)
    s  = pd.Series(r,index=r.keys())
    df = df.append(s,ignore_index=True)
    if count % 1000 ==0:
        print(count)
    count+=1
t_process = time.time() - t1

print('process time (seconds):', t_process)  #takes 8s for 1000
del data

# the above step is slow, so let's write this to csv so we don't have to do it again
df.to_csv('data/phones_processed.csv', index=False)