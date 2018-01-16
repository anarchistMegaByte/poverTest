import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import *

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
a_household_train = pd.read_csv(processed_data_paths['household']['A']['train'], index_col='id')
b_household_train = pd.read_csv(processed_data_paths['household']['B']['train'], index_col='id')
c_household_train = pd.read_csv(processed_data_paths['household']['C']['train'], index_col='id')

a_individual_train = pd.read_csv(processed_data_paths['individual']['A']['train'], index_col='id')
b_individual_train = pd.read_csv(processed_data_paths['individual']['B']['train'], index_col='id')
c_individual_train = pd.read_csv(processed_data_paths['individual']['C']['train'], index_col='id')

#saving aggregated data
a_indiv_agg = a_individual_train.groupby(a_individual_train.index).agg('mean')
b_indiv_agg = b_individual_train.groupby(b_individual_train.index).agg('mean')
c_indiv_agg = c_individual_train.groupby(c_individual_train.index).agg('mean')

def append_i_for_indiv(dataframe):
    dataframe.columns = [str(col) + '_indiv' for col in dataframe.columns]
    return dataframe


#TODO : drop iid fileds before taking it up for post processing
a_indiv_agg = append_i_for_indiv(a_indiv_agg)
a_indiv_agg.to_csv(processed_data_paths['aggregated_indiv']['A']['train'])
b_indiv_agg = append_i_for_indiv(b_indiv_agg)
b_indiv_agg.to_csv(processed_data_paths['aggregated_indiv']['B']['train'])
c_indiv_agg = append_i_for_indiv(c_indiv_agg)
c_indiv_agg.to_csv(processed_data_paths['aggregated_indiv']['C']['train'])

frames_a = [a_household_train, a_indiv_agg]
print("A hhold : " + str(a_household_train.columns.size) + " - indiv : " + str(a_indiv_agg.columns.size))
a_hhold_agg = pd.concat(frames_a,axis=1)
print("A final : " + str(a_hhold_agg.columns.size))
a_hhold_agg.to_csv(processed_data_paths['aggregated_hhold']['A']['train'])

frames_b = [b_household_train, b_indiv_agg]
print("B hhold : " + str(b_household_train.columns.size) + " - indiv : " + str(b_indiv_agg.columns.size))
b_hhold_agg = pd.concat(frames_b,axis=1)
print("B final : " + str(b_hhold_agg.columns.size))
b_hhold_agg.to_csv(processed_data_paths['aggregated_hhold']['B']['train'])

frames_c = [c_household_train, c_indiv_agg]
print("C hhold : " + str(c_household_train.columns.size) + " - indiv : " + str(c_indiv_agg.columns.size))
c_hhold_agg = pd.concat(frames_c,axis=1)
print("C final : " + str(c_hhold_agg.columns.size))
c_hhold_agg.to_csv(processed_data_paths['aggregated_hhold']['C']['train'])
    
 
'''
Treating the test DATA_DIR
'''

# load training data
a_household_test = pd.read_csv(processed_data_paths['household']['A']['test'], index_col='id')
b_household_test = pd.read_csv(processed_data_paths['household']['B']['test'], index_col='id')
c_household_test = pd.read_csv(processed_data_paths['household']['C']['test'], index_col='id')

a_individual_test = pd.read_csv(processed_data_paths['individual']['A']['test'], index_col='id')
b_individual_test = pd.read_csv(processed_data_paths['individual']['B']['test'], index_col='id')
c_individual_test = pd.read_csv(processed_data_paths['individual']['C']['test'], index_col='id')

#saving aggregated data
a_indiv_agg_test = a_individual_test.groupby(a_individual_test.index).agg('mean')
b_indiv_agg_test = b_individual_test.groupby(b_individual_test.index).agg('mean')
c_indiv_agg_test = c_individual_test.groupby(c_individual_test.index).agg('mean')

a_indiv_agg_test = append_i_for_indiv(a_indiv_agg_test)
a_indiv_agg_test.to_csv(processed_data_paths['aggregated_indiv']['A']['test'])
b_indiv_agg_test = append_i_for_indiv(b_indiv_agg_test)
b_indiv_agg_test.to_csv(processed_data_paths['aggregated_indiv']['B']['test'])
c_indiv_agg_test = append_i_for_indiv(c_indiv_agg_test)
c_indiv_agg_test.to_csv(processed_data_paths['aggregated_indiv']['C']['test'])


frames_a_test = [a_household_test, a_indiv_agg_test]
print("A hhold : " + str(a_household_test.columns.size) + " - indiv : " + str(a_indiv_agg_test.columns.size))
a_hhold_agg_test = pd.concat(frames_a_test,axis=1)
print("A final : " + str(a_hhold_agg_test.columns.size))
a_hhold_agg_test.to_csv(processed_data_paths['aggregated_hhold']['A']['test'])

frames_b_test = [b_household_test, b_indiv_agg_test]
print("B hhold : " + str(b_household_test.columns.size) + " - indiv : " + str(b_indiv_agg_test.columns.size))
b_hhold_agg_test = pd.concat(frames_b_test,axis=1)
print("B final : " + str(b_hhold_agg_test.columns.size))
b_hhold_agg_test.to_csv(processed_data_paths['aggregated_hhold']['B']['test'])

frames_c_test = [c_household_test, c_indiv_agg_test]
print("C hhold : " + str(c_household_test.columns.size) + " - indiv : " + str(c_indiv_agg_test.columns.size))
c_hhold_agg_test = pd.concat(frames_c_test,axis=1)
print("C final : " + str(c_hhold_agg_test.columns.size))
c_hhold_agg_test.to_csv(processed_data_paths['aggregated_hhold']['C']['test'])


#master test and train sets

master_train = pd.concat([a_hhold_agg, b_hhold_agg, c_hhold_agg],join='outer')
master_test = pd.concat([a_hhold_agg_test, b_hhold_agg_test, c_hhold_agg_test],join='outer')
master_train.to_csv(processed_data_paths['master']['train'])
master_test.to_csv(processed_data_paths['master']['test'])

to_drop = np.setdiff1d(b_hhold_agg.columns, c_hhold_agg.columns)
#to_common = np.intersect1d(b_hhold_agg.columns, c_hhold_agg.columns)
