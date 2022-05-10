#LOL
#Being able to detect if a team will win or not in the first 15 mins
#Supervised, since it is know which team won each match.
#Need to read into algorithmes for this as I would perfer to use multiple features at ones.
#Presision should be minimum 95 %.
#Do note that this study uses full randomness, nothing is done to ensure even amount of representation between the two teams regarding winning in either the training set or the testing set.
#0 is win for red, 1 is win for blue.

#K-nearest neighbors:
#Test accuracy: 0.5473978459261696
#        precision recall    f1-score   support
#0       0.53      0.77      0.63      6016
#1       0.59      0.33      0.42      6147
#                            0.55     12163 accuracy
#        0.56      0.55      0.53     12163 macro avg
#        0.56      0.55      0.52     12163 weighted avg
#As it can be seen k-nearest neighbors did not solve the problem. 

#Linear SVC
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


from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dfFullSet = pd.read_csv("MatchTimelinesFirst15.csv", index_col=False)
dfSmall = dfFullSet[["redJungleMinionsKilled","blueJungleMinionsKilled","redAvgLevel","blueAvgLevel","redChampKills","blueChampKills","blue_win","redGold", "blueGold", "redMinionsKilled","blueMinionsKilled"]]

x_train, x_test, y_train, y_test = train_test_split(dfSmall[["redJungleMinionsKilled","blueJungleMinionsKilled"]], dfFullSet["blue_win"],
                                                    train_size=0.75, stratify=dfSmall["blue_win"], random_state=5)

fig, ax = plt.subplots(1,2)

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
clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))
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


plt.show()


