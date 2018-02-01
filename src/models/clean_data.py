#### Import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from time import time
import matplotlib.pyplot as plt
from operator import itemgetter
# Load training and test data into pandas dataframes
# ACCCESSING DATA FOLDER REFER THIS LINK 
# (https://stackoverflow.com/questions/7165749/open-file-in-a-relative-location-in-python)

import os

#TO ACCESS THE PATH IN WHICH CURRENT SCRIPT IS RUNNING
fileDir = os.path.dirname(os.path.realpath('__file__'))

#For accessing the file inside a sibling folder.
#data_type = 'raw' or 'processed'
#country   = 'A' or 'B' or 'C'
#person_type = 'hhold' or 'indiv'
#dataset = 'test' or 'train'


def return_file_name(data_type = 'raw', country = 'A', person_type = 'hhold', dataset = 'train'):
    path = 'data/' + data_type + '/' + country + '/' + country + '_' + person_type + '_' + dataset +'.csv'
    print("Building path for file : (" + path +")")
    filename = os.path.join(fileDir, path)
    filename = os.path.abspath(os.path.realpath(filename))
    #print(filename + "\n")
    return filename

def return_train_test(data_type = 'raw', country = 'A', person_type = 'hhold'):
    train = pd.read_csv(return_file_name(data_type = data_type, country = country, person_type = person_type, dataset = 'train'))
    test = pd.read_csv(return_file_name(data_type = data_type, country = country, person_type = person_type, dataset = 'test'))
    # merge training and test sets into one dataframe
    full = pd.concat([train, test])
    return train,test,full
    

def return_bad_cols(dataframe):
    fill_precent_counts = percentage(dataframe.count()-1, dataframe.shape[0]-1)
    bad_cols = fill_precent_counts[fill_precent_counts < 100]
    return bad_cols
    
    
#Looking for Nans
#return a formatted percentage from a fraction
def percentage(numerator, denomenator):
    
    if type(numerator) == pd.core.series.Series:
        return (numerator/denomenator*100)
    
    elif type(numerator) == int or type(numerator) == float:
        return '{:.1f}%'.format(float(numerator)/float(denomenator)*100) 
    
    else:
        print("check type")
#make directory for asving bad columns statistics
# refer link (https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist)
stats_folder_path = 'data/interim/'

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_stats_folder():
    for name in ['A', 'B', 'C']:
        foldername = os.path.join(fileDir, stats_folder_path + name + '/individual')
        foldername = os.path.abspath(os.path.realpath(foldername))
        make_dir(foldername)
        foldername = os.path.join(fileDir, stats_folder_path + name + '/household')
        foldername = os.path.abspath(os.path.realpath(foldername))
        make_dir(foldername)
        print("Trying to create folder for " + name)
        
def return_bad_col_dtype(bad_columns, dataframe):
    data_type = []
    for name in bad_columns:
        data_type.append(dataframe[name].dtype)
    return data_type

create_stats_folder()


# Get size of dataframes
country_name = 'A'

train_set,test_set,full_set = return_train_test(data_type = 'raw', country = country_name, person_type = 'hhold')
train_bad_cols = return_bad_cols(train_set)
test_bad_cols = return_bad_cols(test_set)
test_bad_cols = (test_bad_cols.keys())
train_bad_cols = (train_bad_cols.keys())
train_set_non_na = train_set.copy()
train_set_non_na = train_set_non_na.drop(columns=train_bad_cols,axis=1)
test_set_non_na = test_set.copy()
test_set_non_na = test_set_non_na.drop(columns = test_bad_cols,axis = 1)
train_set_non_na.to_csv(stats_folder_path+ '/A/household/A_hhold_train.csv')
test_set_non_na.to_csv(stats_folder_path+'/A/household/A_hhold_test.csv')


#repeat for indiv
train_set,test_set,full_set = return_train_test(data_type = 'raw', country = country_name, person_type = 'indiv')
train_bad_cols = return_bad_cols(train_set)
test_bad_cols = return_bad_cols(test_set)
test_bad_cols = (test_bad_cols.keys())
train_bad_cols = (train_bad_cols.keys())
train_set_non_na = train_set.copy()
train_set_non_na = train_set_non_na.drop(columns=train_bad_cols,axis=1)
test_set_non_na = test_set.copy()
test_set_non_na = test_set_non_na.drop(columns = test_bad_cols,axis = 1)
train_set_non_na.to_csv(stats_folder_path+ '/A/individual/A_indiv_train.csv')
test_set_non_na.to_csv(stats_folder_path+'/A/individual/A_indiv_test.csv')


