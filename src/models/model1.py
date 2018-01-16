import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import *

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

#code to show the precentage rich /poor people in a family
def create_poor_percentage(data_hhold, data_indiv,name):
    print("Executing for : " + name)
    df_sample = pd.DataFrame(columns=["family_id", "total_persons","rich_ones","poor_ones","percentage_rich","percentage_poor"])

    for familyid in data_hhold.index.unique():
        rich_count = 0
        poor_count = 0
        total_count = 0
        
        for index, row in data_indiv[data_indiv.index == familyid].iterrows():
            total_count += 1
            if row["poor"]:
                poor_count += 1
            else:
                rich_count += 1
            #print(str(row["iid"])+ str(row["poor"]))
        
        df_sample = df_sample.append({
         "family_id": familyid,
         "total_persons": total_count,
         "rich_ones": rich_count,
         "poor_ones": poor_count,
         "percentage_rich": rich_count / total_count,
         "percentage_poor": poor_count / total_count,
          }, ignore_index=True)
    
    df_sample.to_csv('data/processed/' + name +'.csv')



create_poor_percentage(a_household_train, a_individual_train, 'a_hhold_indiv')
create_poor_percentage(b_household_train, b_individual_train, 'b_hhold_indiv')
create_poor_percentage(c_household_train, c_individual_train, 'c_hhold_indiv')

        
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
        df = df.assign(**{c: 0 for c in to_add})
    
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


'''
Build the models
'''
from sklearn.ensemble import RandomForestClassifier
def train_model(features, labels, **kwargs):
    
    # instantiate model
    model = RandomForestClassifier(n_estimators=7, random_state=0)
    
    # train model
    model.fit(features, labels)
    
    # get a (not-very-useful) sense of performance
    accuracy = model.score(features, labels)

    return model
    
model_a = train_model(aX_household_train, ay_household_train)
model_b = train_model(bX_household_train, by_household_train)
model_c = train_model(cX_household_train, cy_household_train)


'''
Predict and Submit
'''
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


#make predictions
a_preds = model_a.predict_proba(a_test)
b_preds = model_b.predict_proba(b_test)
c_preds = model_c.predict_proba(c_test)

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
a_sub = make_country_sub(a_preds, a_test, 'A')
b_sub = make_country_sub(b_preds, b_test, 'B')
c_sub = make_country_sub(c_preds, c_test, 'C')

submission = pd.concat([a_sub, b_sub, c_sub])
submission.to_csv('submission.csv')


'''
write normalised training data in a csv
1) Write A_test , B_test, C_test post processing in a file - 
   /processed/A/A_hhold_test.csv etc.
2) Write aX_train+ay_train in a file
   /processed/A/A_hhold_train.csv etc
'''

#test data
A_hhold_test_pp = a_test.copy()
B_hhold_test_pp = b_test.copy()
C_hhold_test_pp = c_test.copy()
A_indiv_test_pp = aX_test_indiv.copy()
A_indiv_test_pp['iid'] = a_test_indiv.iid
B_indiv_test_pp = bX_test_indiv.copy()
B_indiv_test_pp['iid'] = b_test_indiv.iid
C_indiv_test_pp = cX_test_indiv.copy()
C_indiv_test_pp['iid'] = c_test_indiv.iid

#training data
A_hhold_train_pp = aX_household_train.copy()
A_hhold_train_pp['poor'] = ay_household_train
A_indiv_train_pp = aX_individual_train.copy()
A_indiv_train_pp['poor'] = ay_individual_train
A_indiv_train_pp['iid'] = a_individual_train.iid



B_hhold_train_pp = bX_household_train.copy()
B_hhold_train_pp['poor'] = by_household_train
B_indiv_train_pp = bX_individual_train.copy()
B_indiv_train_pp['poor'] = by_individual_train
B_indiv_train_pp['iid'] = b_individual_train.iid


C_hhold_train_pp = cX_household_train.copy()
C_hhold_train_pp['poor'] = cy_household_train
C_indiv_train_pp = cX_individual_train.copy()
C_indiv_train_pp['poor'] = cy_individual_train
C_indiv_train_pp['iid'] = c_individual_train.iid


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
                    'test':  os.path.join(DATA_DIR, 'C', 'household/C_hhold_test.csv')}}
            }
            
# load training data
A_hhold_train_pp.to_csv(processed_data_paths['household']['A']['train'])
A_indiv_train_pp.to_csv(processed_data_paths['individual']['A']['train'])
B_hhold_train_pp.to_csv(processed_data_paths['household']['B']['train'])
B_indiv_train_pp.to_csv(processed_data_paths['individual']['B']['train'])
C_hhold_train_pp.to_csv(processed_data_paths['household']['C']['train'])
C_indiv_train_pp.to_csv(processed_data_paths['individual']['C']['train'])

#load testing data
A_hhold_test_pp.to_csv(processed_data_paths['household']['A']['test'])
A_indiv_test_pp.to_csv(processed_data_paths['individual']['A']['test'])
B_hhold_test_pp.to_csv(processed_data_paths['household']['B']['test'])
B_indiv_test_pp.to_csv(processed_data_paths['individual']['B']['test'])
C_hhold_test_pp.to_csv(processed_data_paths['household']['C']['test'])
C_indiv_test_pp.to_csv(processed_data_paths['individual']['C']['test'])


'''
get log loss
'''
from sklearn.metrics import log_loss
y_actual = np.concatenate([ay_household_train,by_household_train,cy_household_train],axis = 0)
a_predicted = model_a.predict_proba(aX_household_train)[:,1]
b_predicted = model_b.predict_proba(bX_household_train)[:,1]
c_predicted = model_c.predict_proba(cX_household_train)[:,1]
y_predicted = np.concatenate([a_predicted,b_predicted,c_predicted],axis = 0)

training_score = log_loss(y_actual, y_predicted)
print('training_score =%f'%training_score)







