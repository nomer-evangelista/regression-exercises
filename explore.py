import pandas as pd
import numpy as np

import matplotlib.pylab as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

import wrangle
import env
import explore

def plot_variable_pairs():
    '''
    This function will accepts a dataframe as input and plots all of the pairwise relationships
    Input: train_100
    '''
    train, validate, test = wrangle.wrangle_zillow(wrangle.clean_zillow(wrangle.wrangle_get_zillow()))
    train_100 = train.sample(100)
    sns.pairplot(train_100, corner=True); 
    
    return train_100

    
def plot_categorical_and_continuous_vars():
    '''
    This function will create a visual of displot, boxplot, and regplot
    '''
    
    train, validate, test = wrangle.wrangle_zillow(wrangle.clean_zillow(wrangle.wrangle_get_zillow()))
    train_1000 = train.sample(1000)
    train_con_columns = train[['bedroom', 'bathroom', 'square_ft', 'property_value', 'yr_built', 'tax']]
    train_cat_columns = train[['county']]
    
    train_cat_columns_1000 = train_cat_columns.county.sample(1000)
    train_con_columns_1000 = train_con_columns.sample(1000)

    # Using displot
    sns.displot(data=train_1000, x=train_con_columns_1000.bedroom)
    plt.show();
    
    # Using boxplot
    sns.boxplot(data=train_1000, x=train_con_columns_1000.bathroom, y=train_con_columns_1000.bedroom)
    plt.show();
    
    # Using regplot
    sns.regplot(data=train_1000, x=train_con_columns_1000.bathroom, y=train_con_columns_1000.bedroom, scatter=True)
    plt.show();
    
    # # Using catplot
    # sns.catplot(data=train_1000, x=train_cat_columns_1000.county)
    # plt.show();
    