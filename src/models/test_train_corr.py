import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import *
from scipy.stats.stats import pearsonr 
from scipy.stats.stats import describe

# data directory
# root = 'G:/Old/Courses/poverTest/poverTest'
DATA_DIR = os.path.join('data', 'raw')
'''
LOADING THE DATA
'''
data_paths = { 'individual' : { 'A': {'train': os.path.join(DATA_DIR, 'A', 'A_indiv_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'A', 'A_indiv_test.csv')}, 
              
              'B': {'train': os.path.join(DATA_DIR, 'B', 'B_indiv_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'B', 'B_indiv_test.csv')}, 
              
              'C': {'train': os.path.join(DATA_DIR, 'C', 'C_indiv_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'C', 'C_indiv_test.csv')}},
                    
                'household'  : { 'A': {'train': os.path.join(DATA_DIR, 'A', 'A_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'A', 'A_hhold_test.csv')}, 
              
              'B': {'train': os.path.join(DATA_DIR, 'B', 'B_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'B', 'B_hhold_test.csv')}, 
              
              'C': {'train': os.path.join(DATA_DIR, 'C', 'C_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'C', 'C_hhold_test.csv')}}
            }

# load training data
a_household_train = pd.read_csv(data_paths['household']['A']['train'], index_col='id')
b_household_train = pd.read_csv(data_paths['household']['B']['train'], index_col='id')
c_household_train = pd.read_csv(data_paths['household']['C']['train'], index_col='id')

a_individual_train = pd.read_csv(data_paths['individual']['A']['train'], index_col='id')
b_individual_train = pd.read_csv(data_paths['individual']['B']['train'], index_col='id')
c_individual_train = pd.read_csv(data_paths['individual']['C']['train'], index_col='id')

        
'''
Pre-Process the data
'''
# Standardize features
def standardize(df, numeric_only=True):
    numeric = df.select_dtypes(include=['int64', 'float64'])
    
    # subtracy mean and divide by std
    df[numeric.columns] = (numeric - numeric.mean()) / numeric.std()
    
    return df
    

def pre_process_data(df, enforce_cols=None):
    print("Input shape:\t{}".format(df.shape))
        

    df = standardize(df)
    print("After standardization {}".format(df.shape))
        
    # create dummy variables for categoricals
    df = pd.get_dummies(df)
    print("After converting categoricals:\t{}".format(df.shape))
    

    # match test set and training set columns
    if enforce_cols is not None:
        #Return the sorted, unique values in df.columns that are not in enforce_cols.
        to_drop = np.setdiff1d(df.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, df.columns)

        df.drop(to_drop, axis=1, inplace=True)
        df = df.assign(**{c: np.random.uniform(0,1) for c in to_add})
    
    df.fillna(0, inplace=True)
    
    return df



print("Country A")
aX_household_train = pre_process_data(a_household_train.drop('poor', axis=1))
ay_household_train = np.ravel(a_household_train.poor)
#drop poor and iid as it is individuals id in a family
dropped_poor = a_individual_train.drop('poor', axis=1)
dropped_iid =  dropped_poor.drop('iid', axis=1)
aX_individual_train = pre_process_data(dropped_iid)
ay_individual_train = np.ravel(a_individual_train.poor)


print("\nCountry B")
bX_household_train = pre_process_data(b_household_train.drop('poor', axis=1))
by_household_train = np.ravel(b_household_train.poor)
#drop poor and iid as it is individuals id in a family
dropped_poor = b_individual_train.drop('poor', axis=1)
dropped_iid =  dropped_poor.drop('iid', axis=1)
bX_individual_train = pre_process_data(dropped_iid)
by_individual_train = np.ravel(b_individual_train.poor)


print("\nCountry C")
cX_household_train = pre_process_data(c_household_train.drop('poor', axis=1))
cy_household_train = np.ravel(c_household_train.poor)    
#drop poor and iid as it is individuals id in a family
dropped_poor = c_individual_train.drop('poor', axis=1)
dropped_iid =  dropped_poor.drop('iid', axis=1)
cX_individual_train = pre_process_data(dropped_iid)
cy_individual_train = np.ravel(c_individual_train.poor)    

# load test data
a_test = pd.read_csv(data_paths['household']['A']['test'], index_col='id')
b_test = pd.read_csv(data_paths['household']['B']['test'], index_col='id')
c_test = pd.read_csv(data_paths['household']['C']['test'], index_col='id')

a_test_indiv = pd.read_csv(data_paths['individual']['A']['test'], index_col='id')
b_test_indiv = pd.read_csv(data_paths['individual']['B']['test'], index_col='id')
c_test_indiv = pd.read_csv(data_paths['individual']['C']['test'], index_col='id')


# process the test data
a_test = pre_process_data(a_test, enforce_cols=aX_household_train.columns)
b_test = pre_process_data(b_test, enforce_cols=bX_household_train.columns)
c_test = pre_process_data(c_test, enforce_cols=cX_household_train.columns)

dropped_iid_a =  a_test_indiv.drop('iid', axis=1)
aX_test_indiv = pre_process_data(dropped_iid_a, enforce_cols=aX_individual_train.columns)
dropped_iid_b =  b_test_indiv.drop('iid', axis=1)
bX_test_indiv = pre_process_data(dropped_iid_b, enforce_cols=bX_individual_train.columns)
dropped_iid_c =  c_test_indiv.drop('iid', axis=1)
cX_test_indiv = pre_process_data(dropped_iid_c, enforce_cols=cX_individual_train.columns)



#now we have a_test 4041 x 859
#aX_household_train is 8203 x 859


# corr_array = np.zeros(len(a_household_train))

# max_corr = np.zeros(len(a_test))
# max_corr_index = np.zeros(len(a_test))

# for j in tqdm(range(0,10)):
#     A = a_test.iloc[[j]]
#     for i in range(0,len(corr_array)):
#         B = aX_household_train.iloc[[i]]
#         corr_array[i] = pearsonr(A.T,B.T)[0]
#     max_corr[j] = corr_array.max()
#     max_corr_index[j] = corr_array.argmax()
    
    
# plt.plot(range(0,10),max_corr[0:10])


#try and find correlations between columns in test and train data
#now we have a_test 4041 x 859
#aX_household_train is 8203 x 859

temp = np.shape(bX_household_train)
temp = temp[1]
train_stat = np.zeros((temp,6))
test_stat = np.zeros((temp,6))
feature_corr = np.zeros(temp)
for j in range(0,temp):
    A = bX_household_train[bX_household_train.columns[j]]
    B = b_test[b_test.columns[j]]
    
    train_stat[j,:] = [describe(A).minmax[0], describe(A).minmax[1],describe(A).mean,describe(A).variance,describe(A).skewness, describe(A).kurtosis]
    test_stat[j,:] = [describe(B).minmax[0], describe(B).minmax[1],describe(B).mean,describe(B).variance,describe(B).skewness, describe(B).kurtosis]
        
    feature_corr[j] = (pearsonr(train_stat[j],test_stat[j])[0])

plt.hist(feature_corr, bins = np.arange(-1,1.1,0.1))

sum(feature_corr>0.9)

