#Spam
#Being able to detect if a message is spam or not.
#Got 365 spam and 365 non-spam messages to ensure a balanced traning and testing.
#It was decided to use a supervised method, since the data set used already contains identications and thus a model can easily be trained.
#It was decided to use scikit as the machine learning framework as it could use stochastic gradient decent and support vector machine.  
#The SGD is a linear classifier.
#The support vector mahcine is a set of supervised learning methods, that uses classication, regression, and outliers detection.
#The success rate should be as high as possible, but would, for this study, only be considered good if it had a success rate of 90 %.

#For the first test, 255 rows were selected for training and 110 rows for testing.
#The more training data the better the generalisation will be, but enough testing data is also needed.

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


dfFullSet = pd.read_csv("dataSmall.csv", index_col=False)
dfFullSet["category_encoded"] = preprocessing.LabelEncoder().fit_transform(dfFullSet["category"])

dfSpam = dfFullSet.loc[dfFullSet['category_encoded'] == 1]
dfHam = dfFullSet.loc[dfFullSet['category_encoded'] == 0]
#print(dfSpam)
#print(dfHam)
dfSpamTraining = dfSpam[0:255]
dfHamTraining = dfHam[0:255]
#print(dfSpamTraining)
#print(dfHamTraining)
dfSpamTest = dfSpam[255:366]["message"]
dfHamTest = dfHam[255:366]["message"]
#print(dfSpamTest)
#print(dfHamTest)

trainingDataSet = pd.concat([dfSpamTraining,dfHamTraining])
testDataSet = pd.concat([dfSpamTest,dfHamTest])
print(trainingDataSet)
print(testDataSet)

vectorizer = TfidfVectorizer()
x_train_transformed = vectorizer.fit_transform(trainingDataSet)
x_test_transfomred = vectorizer.transform(testDataSet)
