#LOL
#Being able to detect if a team will win or not in the first 15 mins
#Supervised, since it is known which team won each match.
#Need to read into algorithmes for this as I would perfer to use multiple features at ones.
#Presision should be minimum 90 %.
#Do note that this study uses full randomness, nothing is done to ensure even amount of representation between the two teams regarding winning in either the training set or the testing set.
#0 is win for red, 1 is win for blue.
#The tests are run 75 % of the data for training.

#K-nearest neighbors:
#Test accuracy: 0.5473978459261696
#        precision recall    f1-score   support
#0       0.53      0.77      0.63      6016
#1       0.59      0.33      0.42      6147
#                            0.55     12163 accuracy
#        0.56      0.55      0.53     12163 macro avg
#        0.56      0.55      0.52     12163 weighted avg
#As it can be seen k-nearest neighbors did not solve the problem. 

#Linear SVC:
#Test accuracy: 0.6426046205705829
#        precision recall    f1-score   support
#0       0.64      0.63      0.64      6016
#1       0.64      0.65      0.65      6147
#                            0.64     12163 accuracy
#        0.64      0.64      0.64     12163 macro avg
#        0.64      0.64      0.64     12163 weighted avg
#It does better than k-nearest neighbors, but still not well.

#K-nearest got a higher true positive than Linear SVC, so it is better at estimating a win for red, but it is also much more likely to get a false positive, so it seems to group most matches in the blue_win = 0 class overall.
#Linear SVC got a lower false positive/negative than K-nearest and overall similar scores for true positive and true negative and false positive and false negative.
#The errors were each around 2150 totals and the corrects were around 3900 totals in each, yet it still got around 35-36 % wrong.  

#Linear SGD:
#Test accuracy: 0.6429334868042423
#        precision recall    f1-score   support
#0       0.64      0.63      0.64      6016
#1       0.65      0.65      0.65      6147
#                            0.64     12163 accuracy
#        0.64      0.64      0.64     12163 macro avg
#        0.64      0.64      0.64     12163 weighted avg
#It does better than k-nearest neighbors, but still not well, and is similar to linear SVC.

#The tests have been run with the training data consisting of 95 % of all data and it did not really help. These were for all cases where only red and blue jungle minions killed were looked at.

#The following data is for redAvgLevel, blueAvgLevel, redChampKills, blueChampKills, redGold, blueGold, redMinionsKilled, and blueMiononsKilled
#These have used 75 % of the dataset for training.

#K-nearest neighbors:
#Test accuracy: 0.6986763134095206
#        precision recall    f1-score   support
#0       0.65      0.85      0.74      6016
#1       0.79      0.55      0.65      6147
#                            0.70     12163 accuracy
#        0.72      0.70      0.69     12163 macro avg
#        0.72      0.70      0.69     12163 weighted avg

#Linear SVC:
#Test accuracy: 0.7776042094877909
#        precision recall    f1-score   support
#0       0.78      0.77      0.77      6016
#1       0.78      0.78      0.78      6147
#                            0.78     12163 accuracy
#        0.78      0.78      0.78     12163 macro avg
#        0.78      0.78      0.78     12163 weighted avg

#Linear SGD:
#Test accuracy: 0.776864260462057
#        precision recall    f1-score   support
#0       0.79      0.74      0.77      6016
#1       0.76      0.81      0.79      6147
#                            0.78     12163 accuracy
#        0.78      0.78      0.78     12163 macro avg
#        0.78      0.78      0.78     12163 weighted avg

#As it can be seen these features do a better job at training the model for predicting which team won.
#Interesting enough, K-nearest neighbors is still more likely to predict a match to have been won by the red team.

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dfFullSet = pd.read_csv("MatchTimelinesFirst15.csv", index_col=False)
dfSmall = dfFullSet[["redJungleMinionsKilled","blueJungleMinionsKilled","redAvgLevel","blueAvgLevel","redChampKills","blueChampKills","blue_win","redGold", "blueGold", "redMinionsKilled","blueMinionsKilled"]]
#x_train, x_test, y_train, y_test = train_test_split(dfSmall[["redJungleMinionsKilled","blueJungleMinionsKilled"]], dfFullSet["blue_win"],
#
x_train, x_test, y_train, y_test = train_test_split(dfSmall[["redAvgLevel","blueAvgLevel","redChampKills","blueChampKills","redGold", "blueGold", "redMinionsKilled","blueMinionsKilled"]], dfFullSet["blue_win"],
                                                    train_size=0.75, stratify=dfSmall["blue_win"], random_state=5)

fig, ax = plt.subplots(1,3)

#K-nearest neighbors start
neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(x_train, y_train)
pred_test = neigh.predict(x_test)
pred_train = neigh.predict(x_train)

print("test accuracy", str(np.mean(pred_test == y_test)))
print(classification_report(y_test, pred_test))

cm = confusion_matrix(y_test, pred_test, labels=neigh.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=neigh.classes_)
disp.plot(ax=ax[0])
ax[0].set_title("K-nearest neighbors")
#K-nearest neighbors end

#Linear Support Vector Classication start
clf = make_pipeline(StandardScaler(), LinearSVC( random_state=0, tol=1e-3)) #If running with more features, like the current plus jungle, need to add max_iter=2000 else it will not finish converging in time
clf.fit(x_train,y_train)

pred_test2 = clf.predict(x_test)
pred_train2 = clf.predict(x_train)

print("test accuracy", str(np.mean(pred_test2 == y_test)))
print(classification_report(y_test, pred_test2))

cm = confusion_matrix(y_test, pred_test2, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(ax=ax[1])
ax[1].set_title("Linear SVC")
#Linear Support Vector Classication end


#Linear SGD start
clf2 = make_pipeline(StandardScaler(), SGDClassifier(tol=1e-3, random_state=0))
clf2.fit(x_train,y_train)

pred_test3 = clf2.predict(x_test)
pred_train3 = clf2.predict(x_train)

print("test accuracy", str(np.mean(pred_test3 == y_test)))
print(classification_report(y_test, pred_test3))

cm = confusion_matrix(y_test, pred_test3, labels=clf2.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf2.classes_)
disp.plot(ax=ax[2])
ax[2].set_title("Linear SGD")
#Linear SGD end




plt.show()


