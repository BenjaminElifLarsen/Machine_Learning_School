#Spam
#Being able to detect if a message is spam or not.
#Got 365 spam and 365 non-spam messages to ensure a balanced traning and testing.
#It was decided to use a supervised method, since the data set used already contains identications and thus a model can easily be trained.
#It was decided to use scikit as the machine learning framework as it could use stochastic gradient decent and support vector machine.  
#The SGD is a linear classifier.
#The support vector mahcine is a set of supervised learning methods, that uses classication, regression, and outliers detection.
#The success rate should be as high as possible, but would, for this study, only be considered good if it had a success rate of 90 %.

#For all tests an even amount of rows of ham and spam were selected, e.g. a selection value of 100 indicates 100 spam and 100 ham rows were selected.

#For the first test, 255 rows were selected for training and 110 rows for testing.
#The more training data the better the generalisation will be, but enough testing data is also needed.

#Results for the first test
#Test accuracy: 0.9545
#        precision recall    f1-score   support
#0       0.95      0.96      0.95       110
#1       0.96      0.95      0.95       110
#                            0.95       220 accuracy
#        0.95      0.95      0.95       220 macro avg
#        0.95      0.95      0.95       220 weighted avg
#F1 is combines the precision and recall of a classifer into a single metric, by using their harmonic mean.
#F1 is used to compare performance of a binary classifer
#Precision is how many of instance that were predicted to belong to a specific class actually belonged to it.
#Recall expresses how many instances of a class were predicted correctly.
#Support is how many of each class were present in the testing set.
#From the dataset class 0 is ham, while class 1 is spam
#Thus 5 % of spam messages were false positive, while 4 % of ham messages were false negative.

#For the second test 320 rows were selected for training and 45 rows for testing
#Test accuracy 0.9777777777777777
#        precision recall    f1-score   support
#0       0.98      0.98      0.98        45
#1       0.98      0.98      0.98        45
#                            0.98        90 accuracy
#        0.98      0.98      0.98        90 macro avg
#        0.98      0.98      0.98        90 weighted avg
#As it can be seen the classifer seems to have improved, but it can be argued, and should be, that the test set is most likely to small.

#For the second test 150 rows were selected for training and 215 rows for testing
#        precision recall    f1-score   support
#0       0.94      0.93      0.94       215
#1       0.94      0.94      0.94       215
#                            0.94       430 accuracy
#        0.94      0.94      0.94       430 macro avg
#        0.94      0.94      0.94       430 weighted avg
#As it can be seen, removing 105 rows from the training set did not really devalue the classifer by much.
#Interesting the recall/precision is not mirrowed over the two classes like it was in the other tests.
#After running the test again it ended up being mirrowed, but the values had changed.
#Test accuracy 0.9348837209302325
#        precision recall    f1-score   support
#0       0.95      0.93      0.94       215
#1       0.93      0.95      0.94       215
#                            0.94       430 accuracy
#        0.94      0.94      0.94       430 macro avg
#        0.94      0.94      0.94       430 weighted avg

#This could indicate the training set could benefit from being bigger. However, running some of the other tests again indicates it happens to them to. It is important to notice the same rows are selected in each run of each test.
#This does not change it could be a problem with the size of the training set. In all cases the results are fairly stable with only minor changes in few of the runs.

#An improvement could be to select the test and training data randomly out from the the full dataframe. The reason for this is to get around that some ham and spam might be more, or less, similar to eachothers than other instances.  

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model  import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dfFullSet = pd.read_csv("dataSmall.csv", index_col=False)
dfFullSet["category_encoded"] = preprocessing.LabelEncoder().fit_transform(dfFullSet["category"])
#print(dfFullSet)

dfSpam = dfFullSet.loc[dfFullSet['category_encoded'] == 1]
dfHam = dfFullSet.loc[dfFullSet['category_encoded'] == 0]
#print(dfSpam)
#print(dfHam)

splitValue = 255

dfSpamTraining = dfSpam[0:splitValue]
dfHamTraining = dfHam[0:splitValue]
#print(dfSpamTraining)
#print(dfHamTraining)
dfSpamTest = dfSpam[splitValue:366]["message"]
dfHamTest = dfHam[splitValue:366]["message"]
dfSpamTestCategory = dfSpam[splitValue:366]["category_encoded"]
dfHamTestCategory = dfHam[splitValue:366]["category_encoded"]
#print(dfSpamTest)
#print(dfHamTest)

trainingDataSet = pd.concat([dfSpamTraining,dfHamTraining])
testDataSet = pd.concat([dfSpamTest,dfHamTest])
testDataSetCategory = pd.concat([dfSpamTestCategory,dfHamTestCategory])
#print(trainingDataSet)
#print(testDataSet)

pipeline = Pipeline([("tfidf_vector_com", TfidfVectorizer(
               input="array",
               norm="l2",
               max_features=None,
               sublinear_tf=True)),
          ("clf", SGDClassifier(
               loss="log",
               penalty="l2",
               class_weight="balanced",
               tol=0.001
               ))
          ])
pipeline.fit(trainingDataSet["message"],trainingDataSet["category_encoded"])
pred_test = pipeline.predict(testDataSet)
pred_train = pipeline.predict(trainingDataSet)
print("test accuracy", str(np.mean(pred_test == testDataSetCategory)))
print(classification_report(testDataSetCategory, pred_test))


cm = confusion_matrix(testDataSetCategory, pred_test, labels=pipeline.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
disp.plot()
plt.show()


#How a random selection would be done
#x_train, x_test, y_train, y_test = train_test_split(dfFullSet["message"], dfFullSet["category_encoded"], train_size=splitValue, stratify=dfFullSet["category_encoded"], random_state=5)


