# Bird Song Analysis and Classification.
#
# Features are pre-extracted as chromagrams.
# Identifier is the species of the birds.
# 
# The dataset was pre-preperated into a training and testing set.
# The testing set did not contain entites for all specicies in the training set.
# The training set contained more entities of specific species than other species.
#
# Thus it was decided to:
# - Firstly, try the algorithmes with the prepared datasets.
# - Secondly, to combine the training and testing set and randomly create new sets.
#
# The algoritmes used were:
#
#
#
# The two datasets each consists of 172 columns.
# - One column acted as the label, 'species', of the rows.
# - 13 columns with spectral centroid features, 0 - 12.
#   - The 0 - 12 is the amount of spectrogram frames.
# - 158 columns with chromagram features, ranging from 0 - 11 and 0 - 12. Each bin consists of the normalised energy.
#   - The 0 - 11 is the amount of chroma bins, that is the 12 notes used in the Westen musical scale, that is from C to H.
#   - The 0 - 12 is the amount of spectrogram frames for each chroma bin.
# The training set consists of 1760 rows of birds.
# The testing set consists of 16626 rows of birds. 
#

import pandas as pd
import matplotlib.pyplot as plt

#-- Data Loading -- 
dfTrainingSet = pd.read_csv("train.csv", index_col=False);
dfTestingSet = pd.read_csv("test.csv", index_col=False);


#-- Dataset Presentation --
#pd.set_option('display.max_columns', None)
#print(dfTrainingSet)
#print(dfTestingSet)
#print(dfTrainingSet.describe())
#print(dfTestingSet.describe())


#--  --












