# - Bird Song Analysis and Classification. -

# Features are pre-extracted as chromagrams.
# Identifier is the species of the birds.

# The dataset was pre-preperated into a training and testing set.
# The testing set did not contain entites for all specicies in the training set.
# The training set contained more entities of specific species than other species.
#
# Thus it was decided to:
# - Firstly, try the algorithmes with the prepared datasets.
# - Secondly, to combine the training and testing set and randomly create new sets.

# The algoritmes used were:
# K-nearest neighbours.
# Linear Stochastic Gradient Desent.
# Linear Support Vector Classication.
# Support Vector Classification with Linear Kernal.
#

# The two datasets each consists of 172 columns.
# - One column 'id' which will not be used since it is not a feature nor can be used for labelling.
# - One column 'species', which could act as a label column.
# - One column 'genus' wich could act as a label together with the species column.
# - 13 columns with spectral centroid features, 0 - 12.
#   - The 0 - 12 is the amount of spectrogram frames.
# - 156 columns with chromagram features, ranging from 0 - 11 and 0 - 12. Each bin consists of the normalised energy.
#   - The 0 - 11 is the amount of chroma bins, that is the 12 notes used in the Westen musical scale, that is from C to H.
#   - The 0 - 12 is the amount of spectrogram frames for each chroma bin.
# The training set consists of 1760 rows of birds.
# The testing set consists of 16626 rows of birds. 
# Regarding the columns 'genus' and 'species' there are species, in different genus, that share their names and cases of multiple species under the same genus.
# Thus it was decided to create a new column 'genus_species', consisting of the combine genus and species names, which will be use it for the labelling.


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

#-- Settings --
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None) #This line will display every row and can freeze/crash the program if a lot of rows have to be displayed.


#-- Data Loading -- 
dfTrainingSet = pd.read_csv("train.csv", index_col=False);
dfTestingSet = pd.read_csv("test.csv", index_col=False);
dfTrainingSet.pop("id")
dfTestingSet.pop("id")


#-- Dataset Presentation --
print("DataSet Desciptions:")
print("\nTraining:")
print(dfTrainingSet.describe())
print("\nTesting:")
print(dfTestingSet.describe())
#print(dfTrainingSet.groupby(['genus','species'])['genus','species'].size())
#print(dfTestingSet.groupby(['genus','species'])['genus','species'].size())


#-- Label Column --
label = 'genus_species'
dfTrainingSet[label] = dfTrainingSet["genus"] + " " + dfTrainingSet["species"]
dfTestingSet[label] = dfTestingSet["genus"] + " " + dfTestingSet["species"]
dfTrainingSet.pop("genus")
dfTrainingSet.pop("species")
dfTestingSet.pop("genus")
dfTestingSet.pop("species")
pd.set_option('display.max_rows', None)
print("\nGenus Species:")
print("\nTraining:")
print(*dfTrainingSet[label].unique(),sep='\n')
print("\nTesting:")
print(*dfTestingSet[label].unique(),sep='\n')

# As it can be seen a single bird species, Motacilla flava, is missing from the testing set compared to the training set. 


#-- Get Features --
features = list(dfTrainingSet.columns)
features.remove(label)
print("\nFeatures:")
pd.set_option('display.max_rows', None)
print(*features,sep='\n')


#-- Combined Dataset and Splitting-- 
dfCombined = pd.concat([dfTrainingSet,dfTestingSet])
testPercent = 0.25
randomState = 4
x_train, x_test, y_train, y_test = train_test_split(dfCombined, dfCombined[label], test_size=testPercent, stratify=dfCombined[label], random_state=randomState)
print("\nX_train Genus Species:")
print(*x_train[label].unique(),sep='\n')
print("\nX_test Genus Species:")
print(*x_test[label].unique(),sep='\n')

# As it can be seen here, the training set and the testing set contain the same bird species. 


#-- Label Encoding for Fitting --
le = preprocessing.LabelEncoder()
le.fit(dfCombined[label])
labelEncoded = le.transform(dfCombined[label])
labelEncodedColumnName = label + "_encoded"
dfCombined[labelEncodedColumnName] = le.transform(dfCombined[label])
dfTrainingSet[labelEncodedColumnName] = le.transform(dfTrainingSet[label])
dfTestingSet[labelEncodedColumnName] = le.transform(dfTestingSet[label])
notEncodedLabel = le.inverse_transform(dfCombined[labelEncodedColumnName].unique())
rangeValue = range(len(notEncodedLabel))
print("\nEncoded Genus Species - Genus Species")
for n in rangeValue:
    print(str(n) + " : " + notEncodedLabel[n])

#-- Plot the Datasets --
fig1, ax1 = plt.subplots()
fig1.subplots_adjust(bottom=0.28)
fig1.set_size_inches(20, 10)
dfTrainingSet[label].value_counts().plot.bar(ax=ax1)
plt.setp(ax1.get_xticklabels(), rotation=90)
ax1.set_title("Genus - Species - Training Data")

fig2, ax2 = plt.subplots()
fig2.subplots_adjust(bottom=0.28)
fig2.set_size_inches(20, 10)
dfTestingSet[label].value_counts().plot.bar(ax=ax2)
plt.setp(ax2.get_xticklabels(), rotation=90)
ax2.set_title("Genus - Species - Testing Data")

# As it can seen the training set consists of 20 entities for each species.
# This could be on the low size as some birds have multiple songs and warning sounds compare to others.
# The testing set displays the inbalance clearly, given there are 1711 Alauda Arvensis and only three Perdix Perdix.
# This is a realistic and expected problem as the data is from donated recordings and people are more likely to record songs of specific birds,
# e.g. Alauda Arvensis is much more common bird than Perdix Perdix, while some birds are more likely to sing than others and on different parts of the year. 

fig3, ax3 = plt.subplots()
fig3.subplots_adjust(bottom=0.28)
fig3.set_size_inches(20, 10)
x_train[label].value_counts().plot.bar(ax=ax3)
plt.setp(ax3.get_xticklabels(), rotation=90)
ax3.set_title("Genus - Species - Own Training Data")

fig4, ax4 = plt.subplots()
fig4.subplots_adjust(bottom=0.28)
fig4.set_size_inches(20, 10)
x_test[label].value_counts().plot.bar(ax=ax4)
plt.setp(ax4.get_xticklabels(), rotation=90)
ax4.set_title("Genus - Species - Own Testing Data")


#-- Plot the Heat Maps --
figHeat1, axHeat1 = plt.subplots(1)
figHeat1.subplots_adjust(bottom=0.3)
figHeat1.set_size_inches(40, 40)
dfBird = dfTrainingSet[features]
birdCorr = dfBird.corr(method="spearman")
birdMask = np.triu(np.ones_like(birdCorr, dtype=bool))
sns.heatmap(birdCorr, mask=birdMask, square=True,ax=axHeat1).set(title='Bird Song Features - Training Set')

# The heatmap indicates that there is no correlation between the spectrogram centroids and the chromograms, which is as expected.
# Regarding the chromograms there is overall a high correlation between the frames of the same pitch classes.
# Interesting enough there seems to be a high correlation between a pitch class and the pitch class before it and chromogram 0 correlates hightly with the following chromograms 11, 1, and 2.

figHeat2, axHeat2 = plt.subplots(1)
figHeat2.subplots_adjust(bottom=0.3)
figHeat2.set_size_inches(40, 40)
dfBird2 = dfCombined[features]
birdCorr2 = dfBird2.corr(method="spearman")
birdMask2 = np.triu(np.ones_like(birdCorr2, dtype=bool))
sns.heatmap(birdCorr2, mask=birdMask2, square=True,ax=axHeat2).set(title='Bird Song Features - Full Set')

# The combined dataset does not indicates any real difference in the correlations.


#-- Save Plots --
fig1.savefig('.\\' + ax1.get_title())
plt.close(fig1)
fig2.savefig('.\\' + ax2.get_title())
plt.close(fig2)
fig3.savefig('.\\' + ax3.get_title())
plt.close(fig3)
fig4.savefig('.\\' + ax4.get_title())
plt.close(fig4)

figHeat1.savefig('.\\' + axHeat1.get_title())
plt.close(figHeat1)
figHeat2.savefig('.\\' + axHeat2.get_title())
plt.close(figHeat2)


#-- Classifications Original Training- and Testingset --
print("\nClassifications Original Training- and Testingset")

#--- K-nearest Neighbour ---
neigh = KNeighborsClassifier(n_neighbors=dfTrainingSet[label].unique().size)
neigh.fit(dfTrainingSet[features], dfTrainingSet[labelEncodedColumnName])
pred_test_k = neigh.predict(dfTestingSet[features])
#pred_train_k = neigh.predict(dfTrainingSet[features])

#---- Result ----
print("\nK-Nearest Neighbour")
print("test accuracy", str(np.mean(pred_test_k == dfTestingSet[labelEncodedColumnName])))
print(classification_report(dfTestingSet[labelEncodedColumnName], pred_test_k))

#---- Confusion Matrix ----
figK, axK = plt.subplots(1)
cmK = confusion_matrix(dfTestingSet[labelEncodedColumnName], pred_test_k, labels=neigh.classes_)
dispK = ConfusionMatrixDisplay(confusion_matrix=cmK, display_labels=neigh.classes_)
dispK.plot(ax=axK)
axK.set_title("K-nearest neighbors")


#--- Linear Stochastic Gradient Desent ---
lsgd = make_pipeline(StandardScaler(), SGDClassifier( random_state=randomState, tol=1e-3))
lsgd.fit(dfTrainingSet[features], dfTrainingSet[labelEncodedColumnName])
pred_test_lsgd = lsgd.predict(dfTestingSet[features])

#---- Result ----
print("\nLinear SGD")
print("test accuracy", str(np.mean(pred_test_lsgd == dfTestingSet[labelEncodedColumnName])))
print(classification_report(dfTestingSet[labelEncodedColumnName], pred_test_lsgd))

#---- Confusion Matrix ----
figlsgd, axlsgd = plt.subplots(1)
cmlsgd = confusion_matrix(dfTestingSet[labelEncodedColumnName], pred_test_lsgd, labels=lsgd.classes_)
displsgd = ConfusionMatrixDisplay(confusion_matrix=cmlsgd, display_labels=lsgd.classes_)
displsgd.plot(ax=axlsgd)
axlsgd.set_title("Linear SGD")


#--- Linear Support Vector Classication ---
lsvc = make_pipeline(StandardScaler(), LinearSVC( random_state=randomState, tol=1e-3))
lsvc.fit(dfTrainingSet[features], dfTrainingSet[labelEncodedColumnName])
pred_test_lsvc = lsvc.predict(dfTestingSet[features])

#---- Result ----
print("\nLinear SVC")
print("test accuracy", str(np.mean(pred_test_lsvc == dfTestingSet[labelEncodedColumnName])))
print(classification_report(dfTestingSet[labelEncodedColumnName], pred_test_lsvc))

#---- Confusion Matrix ----
figlsvc, axlsvc = plt.subplots(1)
cmlsvc = confusion_matrix(dfTestingSet[labelEncodedColumnName], pred_test_lsvc, labels=lsvc.classes_)
displsvc = ConfusionMatrixDisplay(confusion_matrix=cmlsvc, display_labels=lsvc.classes_)
displsvc.plot(ax=axlsvc)
axlsvc.set_title("Linear SVC")


#--- Support Vector Classification with Linear Kernal ---
svclk = make_pipeline(StandardScaler(), SVC( kernel="linear", random_state=randomState, tol=1e-3))
svclk.fit(dfTrainingSet[features], dfTrainingSet[labelEncodedColumnName])
pred_test_svclk = svclk.predict(dfTestingSet[features])

#---- Result ----
print("\nSVC Linear Kernel")
print("test accuracy", str(np.mean(pred_test_svclk == dfTestingSet[labelEncodedColumnName])))
print(classification_report(dfTestingSet[labelEncodedColumnName], pred_test_svclk))

#---- Confusion Matrix ----
figsvclk, axsvclk = plt.subplots(1)
cmsvclk = confusion_matrix(dfTestingSet[labelEncodedColumnName], pred_test_svclk, labels=lsvc.classes_)
displaysvclk = ConfusionMatrixDisplay(confusion_matrix=cmsvclk, display_labels=lsvc.classes_)
displaysvclk.plot(ax=axsvclk)
axsvclk.set_title("SVC Linear Kernel")


#--- Gaussian Naive Bayes ---
gnb = make_pipeline(StandardScaler(), GaussianNB())
gnb.fit(dfTrainingSet[features], dfTrainingSet[labelEncodedColumnName])
pred_test_gnb = gnb.predict(dfTestingSet[features])

#---- Result ----
print("\nGaussian Naive Bayes")
print("test accuracy", str(np.mean(pred_test_gnb == dfTestingSet[labelEncodedColumnName])))
print(classification_report(dfTestingSet[labelEncodedColumnName], pred_test_gnb))
                    
#---- Confusion Matrix ----
figgnb, axgnb = plt.subplots(1)
cmgnb = confusion_matrix(dfTestingSet[labelEncodedColumnName], pred_test_gnb, labels=gnb.classes_)
displaygnb = ConfusionMatrixDisplay(confusion_matrix=cmgnb, display_labels=gnb.classes_)
displaygnb.plot(ax=axgnb)
axgnb.set_title("Gaussian Naive Bayes")


#--- Decision Tree Classifier ---
dtc = DecisionTreeClassifier(random_state=randomState)
dtc.fit(dfTrainingSet[features], dfTrainingSet[labelEncodedColumnName])
pred_test_dtc = dtc.predict(dfTestingSet[features])

#---- Result ----
print("\nDecision Tree Classifier")
print("test accuracy", str(np.mean(pred_test_dtc == dfTestingSet[labelEncodedColumnName])))
print(classification_report(dfTestingSet[labelEncodedColumnName], pred_test_dtc))

#---- Confusion Matrix ----
figdtc, axdtc = plt.subplots(1)
cmdtc = confusion_matrix(dfTestingSet[labelEncodedColumnName], pred_test_dtc, labels=dtc.classes_)
displaydtc = ConfusionMatrixDisplay(confusion_matrix=cmdtc, display_labels=dtc.classes_)
displaydtc.plot(ax=axdtc)
axdtc.set_title("Decision Tree Classifier")


#--- Multi-Layer Perceptron Neural Network ---
mlpnn = MLPClassifier(random_state=randomState, max_iter=10000)
mlpnn.fit(dfTrainingSet[features], dfTrainingSet[labelEncodedColumnName])
pred_test_mlpnn = mlpnn.predict(dfTestingSet[features])

#---- Result ----
print("\nMulti-Layer Perceptron Neural Network")
print("test accuracy", str(np.mean(pred_test_mlpnn == dfTestingSet[labelEncodedColumnName])))
print(classification_report(dfTestingSet[labelEncodedColumnName], pred_test_mlpnn))

#---- Confusion Matrix ----
figmlpnn, axmlpnn = plt.subplots(1)
cmmlpnn = confusion_matrix(dfTestingSet[labelEncodedColumnName], pred_test_mlpnn, labels=mlpnn.classes_)
displaymlpnn = ConfusionMatrixDisplay(confusion_matrix=cmmlpnn, display_labels=mlpnn.classes_)
displaymlpnn.plot(ax=axmlpnn)
axmlpnn.set_title("Multi-Layer Perceptron Neural Network")


#--- ---

#---- Result ----

#---- Confusion Matrix ----





#-- Classifications Own Traning- and Testingset --
print("\nClassifications Own Traning- and Testingset")

#--- K-nearest Neighbour ---

#---- Result ----
##print out test accuracy and classication report here

#---- Confusion Matrix ----



#--- Linear Stochastic Gradient Desent ---

#---- Result ----

#---- Confusion Matrix ----



#--- Linear Support Vector Classication ---

#---- Result ----

#---- Confusion Matrix ----



#--- Support Vector Classification with Linear Kernal --- 

#---- Result ----

#---- Confusion Matrix ----



#--- Gaussian Naive Bayes ---

#---- Result ----

#---- Confusion Matrix ----



#--- Decision Tree Classifier ---

#---- Result ----

#---- Confusion Matrix ----



#--- Multi-Layer Perceptron Neural Network ---
mlpnn2 = MLPClassifier(random_state=randomState, max_iter=10000)
mlpnn2.fit(x_train[features], y_train)
pred_test_mlpnn2 = mlpnn2.predict(x_test[features])

#---- Result ----
print("\nMulti-Layer Perceptron Neural Network")
print("test accuracy", str(np.mean(pred_test_mlpnn2 == y_test)))
print(classification_report(y_test, pred_test_mlpnn2))

#---- Confusion Matrix ----



#--- ---

#---- Result ----

#---- Confusion Matrix ----




plt.show()




















