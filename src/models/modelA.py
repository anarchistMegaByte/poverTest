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
b_household_train = pd.read_csv(processed_data_paths['aggregated_hhold']['B']['train'], index_col='id')
b_household_test = pd.read_csv(processed_data_paths['aggregated_hhold']['B']['test'], index_col='id')



# Prepare Data

# Split-out validation dataset
dataset = b_household_train.copy()
dataset = dataset.drop('poor', axis=1)
array = dataset.values
X = array.astype(float)
Y = b_household_train['poor']
validation_size = 0.20
seed = 7
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=validation_size, random_state=seed)

features = dataset.columns
clf = RandomForestClassifier(random_state = 84)


# function takes a RF parameter and a ranger and produces a plot and dataframe of CV scores for parameter values
def evaluate_param(parameter, num_range, index):
    grid_search = GridSearchCV(clf, param_grid = {parameter: num_range})
    grid_search.fit(X_train, y_train)
    
    df = {}
    for i, score in enumerate(grid_search.grid_scores_):
        df[score[0][parameter]] = score[1]
       
    
    df = pd.DataFrame.from_dict(df, orient='index')
    df.reset_index(level=0, inplace=True)
    df = df.sort_values(by='index')
 
    plt.subplot(3,2,index)
    plot = plt.plot(df['index'], df[0])
    plt.title(parameter)
    return plot, df

# parameters and ranges to plot
param_grid = {"n_estimators": np.arange(2, 300, 2),
              "max_depth": np.arange(1, 28, 1),
              "min_samples_split": np.arange(0.1,0.5,0.1).astype(float),
              "min_samples_leaf": np.arange(0.1,0.5,0.1),
              "max_leaf_nodes": np.arange(2,60,1),
              "min_weight_fraction_leaf": np.arange(0.1,0.4, 0.1)}

index = 1
plt.figure(figsize=(16,12))
for parameter, param_range in dict.items(param_grid):   
    evaluate_param(parameter, param_range, index)
    index += 1


plot = plt.plot(df['index'], df[0])
plt.title(parameter)




'''# Evaluate Algorithms

# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'accuracy'

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

'''






