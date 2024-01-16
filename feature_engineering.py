import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

import wrangle
import env
import explore

def scaled_version(train, validate, test, scaler, columns_to_scale, return_scaler=False):
    '''
    This function will intake train, validate, and test that is already been split and 
    returns a copy of scaled version of scaler
    
    Example columns to scale: columns_to_scale=['bedroom', 'bathroom', 'tax', 'square_ft']
    
    Input: scaler = train, validate, test, scaler = MinMaxScaler(), StandardScaler(), RobustScaler()
    and/or return_scaler=True
    
    Output: save to variable = train_scaled, validate_scaled, test_scaled, and/or return_scaler=True
    '''
    # making a copy of df 
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    # creating the scaler 
    scaler = scaler
    # fit the train df
    scaler.fit(train[columns_to_scale])
   
    # applying scaler to transform df 
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                    columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),                                                   columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                    columns=test[columns_to_scale].columns.values).set_index([test.index.values])                                                                                  
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    
    else:
        return train_scaled, validate_scaled, test_scaled