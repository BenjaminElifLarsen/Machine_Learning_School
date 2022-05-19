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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
dfTrainingSet["genus_species"] = dfTrainingSet["genus"] + " " + dfTrainingSet["species"]
dfTestingSet["genus_species"] = dfTestingSet["genus"] + " " + dfTestingSet["species"]
dfTrainingSet.pop("genus")
dfTrainingSet.pop("species")
dfTestingSet.pop("genus")
dfTestingSet.pop("species")
pd.set_option('display.max_rows', None)
print("\nGenus Species:")
print("\nTraining:")
print(*dfTrainingSet["genus_species"].unique(),sep='\n')
print("\nTesting:")
print(*dfTestingSet["genus_species"].unique(),sep='\n')

# As it can be seen a single bird species, Motacilla flava, is missing from the testing set compared to the training set. 


#-- Get Features --
features = list(dfTrainingSet.columns)
features.remove("genus_species")
print("\nFeatures:")
pd.set_option('display.max_rows', None)
print(*features,sep='\n')


#-- Combined Dataset and Splitting-- 
dfCombined = pd.concat([dfTrainingSet,dfTestingSet])
testPercent = 0.5
randomState = 4
x_train, x_test, y_train, y_test = train_test_split(dfCombined, dfCombined["genus_species"], test_size=testPercent, stratify=dfCombined["genus_species"], random_state=randomState)
print("\nX_train Genus Species:")
print(*x_train["genus_species"].unique(),sep='\n')
print("\nX_test Genus Species:")
print(*x_test["genus_species"].unique(),sep='\n')

# As it can be seen here, the training set and the testing set contain the same bird species. 


#-- Plot the Datasets --
fig1, ax1 = plt.subplots()
fig1.subplots_adjust(bottom=0.28)
fig1.set_size_inches(20, 10)
dfTrainingSet["genus_species"].value_counts().plot.bar(ax=ax1)
plt.setp(ax1.get_xticklabels(), rotation=90)
ax1.set_title("Genus - Species - Training Data")

fig2, ax2 = plt.subplots()
fig2.subplots_adjust(bottom=0.28)
fig2.set_size_inches(20, 10)
dfTestingSet["genus_species"].value_counts().plot.bar(ax=ax2)
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
x_train["genus_species"].value_counts().plot.bar(ax=ax3)
plt.setp(ax3.get_xticklabels(), rotation=90)
ax3.set_title("Genus - Species - Own Training Data")

fig4, ax4 = plt.subplots()
fig4.subplots_adjust(bottom=0.28)
fig4.set_size_inches(20, 10)
x_test["genus_species"].value_counts().plot.bar(ax=ax4)
plt.setp(ax4.get_xticklabels(), rotation=90)
ax4.set_title("Genus - Species - Own Testing Data")



#-- Heat Map --
figHeat, axHeat = plt.subplots(1)
figHeat.subplots_adjust(bottom=0.3)
figHeat.set_size_inches(10, 40)
dfBird = dfTrainingSet[features]
birdCorr = dfBird.corr(method="spearman")
birdMask = np.triu(np.ones_like(birdCorr, dtype=bool))
sns.heatmap(birdCorr, mask=birdMask, square=True,ax=axHeat).set(title='Bird Song Features - Training Set')

# The heatmap indicates that there is no correlation between the spectrogram centroids and the chromograms, which is as expected.
# Regarding the chromograms there is overall a high correlation between the frames of the same pitch classes.
# Interesting enough there seems to be a high correlation between a pitch class and the pitch class before it and chromogram 0 correlates hightly with the following chromograms 11, 1, and 2.


figHeat2, axHeat2 = plt.subplots(1)
figHeat2.subplots_adjust(bottom=0.3)
figHeat2.set_size_inches(10, 40)
dfBird2 = dfCombined[features]
birdCorr2 = dfBird2.corr(method="spearman")
birdMask2 = np.triu(np.ones_like(birdCorr2, dtype=bool))
sns.heatmap(birdCorr2, mask=birdMask2, square=True,ax=axHeat2).set(title='Bird Song Features- Full Set')

# The combined dataset does not indicates any real difference in the correlations.


#-- Classifications --

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



#--- ---

#---- Result ----

#---- Confusion Matrix ----



#--- ---

#---- Result ----

#---- Confusion Matrix ----



#--- ---

#---- Result ----

#---- Confusion Matrix ----



#--- ---

#---- Result ----

#---- Confusion Matrix ----





#plt.tight_layout()

plt.show()




















