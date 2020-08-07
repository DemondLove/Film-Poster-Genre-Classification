# Import libraries
import numpy as np
import pandas as pd
from keras.utils import to_categorical

def filterNULLs(df):
    '''
    Add DataFrame column with the Primary Genre associated with each film

    Parameters:
    df (pd.DataFrame) - Input DataFrame with Genre column containing pipe delimited list of genres associated with each film

    Returns:
    df (pd.DataFrame) - Updated DataFrame
    '''
    print('src.dataPrep.filterNULLs')

    for col in df.columns:
        if df[col].isna().sum() > 0:
            df = df[df[col].notna()]

    return df

def extractPrimaryGenre(df):
    '''
    Add DataFrame column with the Primary Genre associated with each film

    Parameters:
    df (pd.DataFrame) - Input DataFrame with Genre column containing pipe delimited list of genres associated with each film

    Returns:
    df (pd.DataFrame) - Updated DataFrame
    '''
    print('src.dataPrep.extractPrimaryGenre')

    primaryGenre = []

    df['primaryGenre'] = df['Genre'].str.split('|')

    for index, row in df.iterrows():
        primaryGenre.append(row['primaryGenre'][0])

    df['primaryGenre'] = primaryGenre

    return df

def filterGenres(df):
    '''
    Remove rows that are associated to genres that don't contain at least 5% of the distribution

    Parameters:
    df (pd.DataFrame) - Input DataFrame with PrimaryGenre column containing the genre associated with each film

    Returns:
    df (pd.DataFrame) - Updated DataFrame
    '''
    print('src.dataPrep.filterGenres')

    df['genreCount'] = df.groupby('primaryGenre')['primaryGenre'].transform('count')

    genreCnt = len(df)*.05

    return df.query('genreCount > @genreCnt')

def convertGenreVariable(df):
    '''
    Convert primaryGenre column to numeric variable

    Parameters:
    df (pd.DataFrame) - Input DataFrame with primaryGenre column containing genre associated with each film

    Returns:
    primaryGenre (np.numpy) - Numpy Array of nested arrays that are used for target variable
    '''
    print('src.dataPrep.convertGenreVariable')

    df['primaryGenreCode'] = df['primaryGenre'].replace(list(df['primaryGenre'].unique())
                                                        , list(range(len(df['primaryGenre'].unique()))))

    primaryGenre = to_categorical(np.array(df['primaryGenreCode'])
                                , num_classes=6)

    return primaryGenre