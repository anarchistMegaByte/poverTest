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


from sklearn.decomposition import PCA, KernelPCA
pca = KernelPCA(n_components=2,kernel = 'poly')
pca.fit(aX_household_train)
ax_pca = pca.transform(aX_household_train)

pca.fit(bX_household_train)
bx_pca = pca.transform(bX_household_train)

pca.fit(cX_household_train)
cx_pca = pca.transform(cX_household_train)

plt.figure(figsize=(8,6))
plt.scatter(ax_pca[:,0],(ax_pca[:,1]),c=ay_household_train,cmap='plasma')
plt.xlabel('First principal component for A Household')
plt.ylabel('Second Principal Component for A Household')

plt.figure(figsize=(8,6))
plt.scatter(bx_pca[:,0],bx_pca[:,1],c=by_household_train,cmap='plasma')
plt.xlabel('First principal component for B Household')
plt.ylabel('Second Principal Component for B Household')

plt.figure(figsize=(8,6))
plt.scatter(cx_pca[:,0],cx_pca[:,1],c=cy_household_train,cmap='plasma')
plt.xlabel('First principal component for C Household')
plt.ylabel('Second Principal Component for C Household')


