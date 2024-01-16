import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

import wrangle
import env
import explore

def visualize_scaler(scaler, df, columns_to_scale, bins=10):
    '''
    This function will output a visualize before and after scaling
    
    Input: Scaler = MinMaxScaler(), StandardScaler(), RobustScaler(), 
    df, columns_to_scale, bins=
    
    Output: None
    '''
    fig, axs = plt.subplots(len(columns_to_scale), 2, figsize=(16,9))
    df_scaled = df.copy()
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    for (ax1, ax2), col in zip(axs, columns_to_scale):
        ax1.hist(df[col], bins=bins)
        ax1.set(title=f'{col} before scaling', xlabel=col, ylabel='count')
        ax2.hist(df_scaled[col], bins=bins)
        ax2.set(title=f'{col} after scaling with {scaler.__class__.__name__}', xlabel=col, ylabel='count')
    plt.tight_layout()
    
    
def robust_scaler(df, cols=None):
    """
    Applies a RobustScaler to all numeric columns in a pandas DataFrame.

    Parameters:
    df (pandas.DataFrame): The input dataframe.

    Returns:
    pandas.DataFrame: The input dataframe with scaled numeric columns.
    """
    if cols == None:
        # Select only the numeric columns
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

        # Create a RobustScaler object
        scaler = RobustScaler()

        # Scale the numeric columns and create a new DataFrame
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        # Create a RobustScaler object
        scaler = RobustScaler()

        # Scale the numeric columns and create a new DataFrame
        df[cols] = scaler.fit_transform(df[cols])

    # Return the modified dataframe
    return df


def standard_scaler(df, cols):
    """
    Scales the columns in a dataframe using StandardScaler.
    Takes in a dataframe and a list of columns to scale.
    Returns the scaled dataframe.
    """
    # create an instance of the StandardScaler class
    scaler = StandardScaler()

    # fit the scaler to the columns to be scaled
    scaler.fit(df[cols])

    # transform the columns to be scaled using the fitted scaler
    scaled_cols = scaler.transform(df[cols])

    # create a new dataframe with the scaled columns
    scaled_df = pd.DataFrame(scaled_cols, columns=cols, index=df.index)

    # replace the original columns with the scaled columns in the original dataframe
    df[cols] = scaled_df[cols]

    return df


def min_max_scaler(df, cols):
    """
    Scales the columns in a dataframe using MinMaxScaler.
    Takes in a dataframe and a list of columns to scale.
    Returns the scaled dataframe.
    """
    # create an instance of the MinMaxScaler class
    scaler = MinMaxScaler()

    # fit the scaler to the columns to be scaled
    scaler.fit(df[cols])

    # transform the columns to be scaled using the fitted scaler
    scaled_cols = scaler.transform(df[cols])

    # create a new dataframe with the scaled columns
    scaled_df = pd.DataFrame(scaled_cols, columns=cols, index=df.index)

    # replace the original columns with the scaled columns in the original dataframe
    df[cols] = scaled_df[cols]

    return df

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
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                    columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                    columns=test[columns_to_scale].columns.values).set_index([test.index.values])
                                                  
                                                  
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    
    else:
        return train_scaled, validate_scaled, test_scaled
    