# Import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from time import time
import matplotlib.pyplot as plt
from operator import itemgetter
from datacleaner import autoclean
from datacleaner import autoclean_cv

# Load training and test data into pandas dataframes
input_path = 'data/interim/C/'
output_path = 'data/processed/C/'
local_path = 'individual/'
train = pd.read_csv(input_path+ local_path + 'C_indiv_train.csv')
test = pd.read_csv(input_path + local_path+ 'C_indiv_test.csv')


clean_train = autoclean(train)
clean_test = autoclean(test)

clean_train.to_csv(output_path+local_path+'C_indiv_train.csv')
clean_test.to_csv(output_path+local_path+'C_indiv_test.csv')



'''
# merge training and test sets into one dataframe
full = pd.concat([train, test])


#return a formatted percentage from a fraction
def percentage(numerator, denomenator):
    if type(numerator) == pd.core.series.Series:
        return (numerator/denomenator*100)
    
    elif type(numerator) == int or type(numerator) == float:
        return '{:.1f}%'.format(float(numerator)/float(denomenator)*100) 
    
    else:
        print("check type")
        
        
def check_nan_values(full):
    A = percentage(full.count()-1, full.shape[0]-1)
    A_bad = A[A<100]
    bad_column_index = np.argwhere(A_bad == True)
    return (A_bad)
        

def fill_na_values(df type_s):
    if type_s == 'most_freq':
        bad_column = check_nan_values(df)
        temp_notnull = df[bad_column.keys()[0]].dropna()
        most_freq_val = np.histogram(temp_notnull)[1][np.argmax(np.histogram(temp_notnull)[0])]
        df = df[bad_column.keys()[0]].fillna(most_freq_val,inplace = True)    
        return df
    
#Get percentage by variable of values which are not NaN
bad_column = check_nan_values(full)

temp_notnull = full[bad_column.keys()[0]].dropna()
most_freq_val = np.histogram(temp_notnull)[1][np.argmax(np.histogram(temp_notnull)[0])]

#todo - investigate further if replacing with most freuent value is good approach
full[bad_column.keys()[0]].fillna(most_freq_val,inplace = True)
'''