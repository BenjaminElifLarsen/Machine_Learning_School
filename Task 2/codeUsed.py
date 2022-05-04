import pandas as pd

fileLines = []
with open('SMSSpamCollection','r') as f:
  for line in f:
    fl = f.readline()
    fl = fl.replace('\n','')
    fl = fl.split('\t',1)
    fileLines.append(fl)

df = pd.DataFrame(fileLines, columns=['category','message'])
df.to_csv('data.csv')

percent = df['category'].value_counts(normalize=True)*100
print(percent)
dfHam = df.loc[df['category'] == 'ham']
dfSpam = df.loc[df['category'] == 'spam']
print(round((dfSpam.size/dfHam.size),2))
dfNoIndex = dfHam.append(dfHam, ignore_index=True)
rows = dfNoIndex.loc[dfNoIndex.index < dfSpam.size/2]
frames = [dfSpam, rows]
result = pd.concat(frames)
oneHot = pd.get_dummies(result.category)
result.to_csv('dataSmall.csv')
