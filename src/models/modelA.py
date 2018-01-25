import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import *
import seaborn as sns


# Classification Project: Sonar rocks or mines

# Load libraries
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

#read data into data frame
DATA_DIR = os.path.join('data', 'processed')
processed_data_paths = { 'individual' : { 'A': {'train': os.path.join(DATA_DIR, 'A', 'individual/A_indiv_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'A', 'individual/A_indiv_test.csv')}, 
              
              'B': {'train': os.path.join(DATA_DIR, 'B', 'individual/B_indiv_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'B', 'individual/B_indiv_test.csv')}, 
              
              'C': {'train': os.path.join(DATA_DIR, 'C', 'individual/C_indiv_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'C', 'individual/C_indiv_test.csv')}},
                    
                'household'  : { 'A': {'train': os.path.join(DATA_DIR, 'A', 'household/A_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'A', 'household/A_hhold_test.csv')}, 
              
              'B': {'train': os.path.join(DATA_DIR, 'B', 'household/B_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'B', 'household/B_hhold_test.csv')}, 
              
              'C': {'train': os.path.join(DATA_DIR, 'C', 'household/C_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'C', 'household/C_hhold_test.csv')}},
                
                'aggregated_indiv'  : { 'A': {'train': os.path.join(DATA_DIR, 'A', 'individual/A_indiv_aggre_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'A', 'individual/A_indiv_aggre_test.csv')}, 
              
              'B': {'train': os.path.join(DATA_DIR, 'B', 'individual/B_indiv_aggre_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'B', 'individual/B_indiv_aggre_test.csv')}, 
              
              'C': {'train': os.path.join(DATA_DIR, 'C', 'individual/C_indiv_aggre_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'C', 'individual/C_indiv_aggre_test.csv')}},
                
                'aggregated_hhold'  : { 'A': {'train': os.path.join(DATA_DIR, 'aggregated', 'A_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'aggregated', 'A_test.csv')}, 
              
              'B': {'train': os.path.join(DATA_DIR, 'aggregated', 'B_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'aggregated', 'B_test.csv')}, 
              
              'C': {'train': os.path.join(DATA_DIR, 'aggregated', 'C_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'aggregated', 'C_test.csv')}},
                
                'master' : { 'train' : os.path.join(DATA_DIR, 'aggregated', 'master_train.csv'),
                             'test' : os.path.join(DATA_DIR, 'aggregated', 'master_test.csv')}
            }


'''
Treating the training DATA_DIR
'''

# load training data
a_household_train = pd.read_csv(processed_data_paths['aggregated_hhold']['A']['train'], index_col='id')
a_household_test = pd.read_csv(processed_data_paths['aggregated_hhold']['A']['test'], index_col='id')



# Prepare Data

# Split-out validation dataset
dataset = a_household_train.copy()
dataset = dataset.drop('poor', axis=1)
array = dataset.values
X = array.astype(float)
Y = a_household_train['poor']
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Evaluate Algorithms

# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'accuracy'

'''
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# Standardize the dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression())])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC())])))
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
'''

# ensembles
ensembles = []
ensembles.append(('AB', AdaBoostClassifier()))
ensembles.append(('GBM', GradientBoostingClassifier()))
ensembles.append(('RF', RandomForestClassifier()))
ensembles.append(('ET', ExtraTreesClassifier()))
results = []
names = []
for name, model in ensembles:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()








