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

a_household_train = pd.read_csv(processed_data_paths['aggregated_hhold']['A']['train'], index_col='id')
a_household_test = pd.read_csv(processed_data_paths['aggregated_hhold']['A']['test'], index_col='id')

c_household_train = pd.read_csv(processed_data_paths['aggregated_hhold']['C']['train'], index_col='id')
c_household_test = pd.read_csv(processed_data_paths['aggregated_hhold']['C']['test'], index_col='id')


# Prepare Data

# Split-out validation dataset
dataset_a = a_household_train.copy()
dataset_a = dataset_a.drop('poor', axis=1)
dataset_a = dataset_a.drop('Unnamed: 0',axis=1)
array = dataset_a.values
X_a= array.astype(float)
Y_a = a_household_train['poor']
validation_size = 0.20
seed = 7
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_a, Y_a, test_size=validation_size, random_state=seed)

dataset_b = b_household_train.copy()
dataset_b = dataset_b.drop('poor', axis=1)
dataset_b = dataset_b.drop('Unnamed: 0',axis=1)
array = dataset_b.values
X_b= array.astype(float)
Y_b = b_household_train['poor']
validation_size = 0.20
seed = 7
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_b, Y_b, test_size=validation_size, random_state=seed)

dataset_c = c_household_train.copy()
dataset_c = dataset_c.drop('poor', axis=1)
dataset_c = dataset_c.drop('Unnamed: 0',axis=1)
array = dataset_c.values
X_c = array.astype(float)
Y_c = c_household_train['poor']
validation_size = 0.20
seed = 7
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, Y_c, test_size=validation_size, random_state=seed)

classifier_a = RandomForestClassifier(n_estimators=25,min_samples_split=0.3,max_depth=10,min_weight_fraction_leaf= 0.2,max_leaf_nodes= 20)
classifier_b = RandomForestClassifier(n_estimators=55,min_samples_split=0.3,max_depth=10,max_leaf_nodes= 40)
classifier_c = RandomForestClassifier(n_estimators=25,min_samples_split=0.25,max_depth=1,max_leaf_nodes= 15)


#make predictions
classifier_a.fit(X_train_a,y_train_a)
a_preds = classifier_a.predict_proba(a_household_test)

classifier_b.fit(X_train_b,y_train_b)
b_preds = classifier_b.predict_proba(b_household_test)

classifier_c.fit(X_train_c,y_train_c)
c_preds = classifier_c.predict_proba(c_household_test)

def make_country_sub(preds, test_feat, country):
    # make sure we code the country correctly
    country_codes = ['A', 'B', 'C']
    
    # get just the poor probabilities
    country_sub = pd.DataFrame(data=preds[:, 1],  # proba p=1
                               columns=['poor'], 
                               index=test_feat.index)

    
    # add the country code for joining later
    country_sub["country"] = country
    return country_sub[["country", "poor"]]
    
# convert preds to data frames
a_sub = make_country_sub(a_preds, a_household_test, 'A')
b_sub = make_country_sub(b_preds, b_household_test, 'B')
c_sub = make_country_sub(c_preds, c_household_test, 'C')

a_test_temp = read_csv('data/raw/A/' + 'a_hhold_test.csv')
b_test_temp = read_csv('data/raw/B/' + 'b_hhold_test.csv')
c_test_temp = read_csv('data/raw/C/' + 'c_hhold_test.csv')

a_sub = a_sub.reindex(a_test_temp.id)
b_sub = b_sub.reindex(b_test_temp.id)
c_sub = c_sub.reindex(c_test_temp.id)

submission = pd.concat([a_sub, b_sub, c_sub])
submission.to_csv('submission.csv')



