import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("MatchTimelinesFirst15.csv",index_col=False)


dfSmall = df[["redAvgLevel","blueAvgLevel","redChampKills","blueChampKills","blue_win","redGold", "blueGold", "redMinionsKilled","blueMinionsKilled"]]

#sns.boxplot(x=dfSmall["blue_win"], y=dfSmall["redAvgLevel"])
#sns.boxplot(x=dfSmall["blue_win"], y=dfSmall["blueAvgLevel"])
#sns.boxplot(x=dfSmall["blue_win"], y=dfSmall["redChampKills"])
#sns.boxplot(x=dfSmall["blue_win"], y=dfSmall["blueChampKills"])

dfTest = pd.DataFrame(data=dfSmall, columns=["redAvgLevel","blueAvgLevel", "blue_win"])

#print(dfTest)
data=dfTest.melt(dfTest)
#print(data)
#sns.boxplot(x=dfTest["blue_win"], y="value", data=data)
sns.boxplot(data=dfTest)
plt.show()
