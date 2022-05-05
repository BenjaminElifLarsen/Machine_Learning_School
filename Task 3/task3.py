import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("MatchTimelinesFirst15.csv",index_col=False)


dfSmall = df[["redJungleMinionsKilled","blueJungleMinionsKilled","redAvgLevel","blueAvgLevel","redChampKills","blueChampKills","blue_win","redGold", "blueGold", "redMinionsKilled","blueMinionsKilled"]]

##sns.boxplot(x=dfSmall["blue_win"], y=dfSmall["redAvgLevel"])
#sns.boxplot(x=dfSmall["blue_win"], y=dfSmall["blueAvgLevel"])
#sns.boxplot(x=dfSmall["blue_win"], y=dfSmall["redChampKills"])
#sns.boxplot(x=dfSmall["blue_win"], y=dfSmall["blueChampKills"])

dfTest = pd.DataFrame(data=dfSmall, columns=["redAvgLevel","blueAvgLevel", "blue_win"])

#print(dfTest)
data=dfTest.melt(dfTest)
#print(data)
#sns.boxplot(x=dfTest["blue_win"], y="value", data=data)

fig2, ax2 = plt.subplots(1,2)
dfRed = dfSmall[["redJungleMinionsKilled","redAvgLevel","redChampKills","redGold","redMinionsKilled"]]
redCorr = dfRed.corr(method="spearman")
redMask = np.triu(np.ones_like(redCorr, dtype=bool))
dfBlue = dfSmall[["blueJungleMinionsKilled","blueAvgLevel","blueChampKills","blueGold","blueMinionsKilled"]]
blueCorr = dfBlue.corr(method="spearman")
blueMask = np.triu(np.ones_like(blueCorr, dtype=bool))

sns.heatmap(redCorr, mask=redMask, annot=True, square=True,ax=ax2[0]).set(title='Red Team')
sns.heatmap(blueCorr, mask=blueMask, annot=True, square=True,ax=ax2[1]).set(title='Blue Team')


fig, ax = plt.subplots(3,1)
dfBlueWin = dfTest.loc[dfTest['blue_win'] == 1]
dfRedWin = dfTest.loc[dfTest['blue_win'] == 0]
h1 = sns.histplot(data=dfBlueWin[["redAvgLevel","blueAvgLevel"]], color=['r','b'], palette=["red",'blue'], shrink=8, multiple="dodge", ax=ax[0]).set(title='Blue Win',xlim=(1,12))
h2 = sns.histplot(data=dfRedWin[["redAvgLevel","blueAvgLevel"]], color=['r','b'], palette=["red",'blue'], shrink=8, multiple="dodge", ax=ax[1]).set(title='Red Win',xlim=(1,12))
h3 = sns.histplot(data=dfTest[["redAvgLevel","blueAvgLevel"]], color=['r','b'], palette=["red",'blue'], shrink=8, multiple="dodge", ax=ax[2]).set(title='Both Wins',xlim=(1,12))


#g = sns.catplot(col="blue_win",data=dfTest, kind="box", palette=["red",'blue'], height=4, aspect=.7)

plt.show()
