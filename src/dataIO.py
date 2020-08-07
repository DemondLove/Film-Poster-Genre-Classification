# Import libraries
import numpy as np
import pandas as pd
import requests
from PIL import Image
from keras.preprocessing.image import img_to_array


def readCSV(path, file_name, encoding):
    '''
    Wrapper for Pandas read_csv

    Parameters:
    path (str) - network path to the file
    file_name (str) - name of file, including extension
    encoding (str) - file encoding

    Returns:
    df (pd.DataFrame) - csv DataFrame
    '''
    print('src.dataIO.readCSV')

    try:
        df = pd.read_csv(path+file_name, encoding=encoding)

        return df
    except:
        return 'File Not Found'


def downloadPosterArray(df):
    '''
    Download jpeg files of Film Posters online, convert to Numpy Array, and update DataFrames
    Only keep Posters converted to (268, 182, 3)

    Parameters:
    df (pd.DataFrame) - DataFrame including Poster of image urls & imdbId of unique identifiers

    Returns:
    df (pd.DataFrame) - Input DataFrame with all rows removed that don't correspond to 3D Poster images
    posterArray (np.array) - numpy arrays of the 3D Poster images
    '''
    print('src.dataIO.downloadPosterArray')

    idx = []
    posterArray = []

    for index, row in df.iterrows():

        try:
            tempArray = img_to_array(Image.open(requests.get(row['Poster'], stream=True).raw))

            if tempArray.shape != (268, 182, 3):
                idx.append(row['imdbId'])

            else:
                posterArray.append(tempArray)

        except:
            idx.append(row['imdbId'])

    df = df.query('imdbId not in @idx')
    posterArray = np.array(posterArray)

    return df, posterArray

def saveDatasets(df, posterArray, file_name_df, file_name_posterArray):
    '''
    Save updated dataset as .csv and .npy

    Parameters:
    df (pd.DataFrame) - DataFrame containing target variable
    posterArray (np.array) - 3D numpy array of poster image data used as independent variable
    file_name_df (str) - Name of the csv file created
    file_name_posterArray (str) - Name of the npy file created
    '''
    print('src.dataIO.saveDatasets')

    df.to_csv('data/'+file_name_df+'.csv', index=False)
    np.save('data/'+file_name_posterArray, posterArray)

def readSavedDatasets(file_name_df, file_name_posterArray):
    '''
    Read saved .csv and .npy datasets

    Parameters:
    file_name_df (str) - Name of the csv file
    file_name_posterArray (str) - Name of the npy file

    Returns:
    df (pd.DataFrame) - DataFrame containing target variable
    posterArray (np.array) - 3D numpy array of poster image data used as independent variable
    '''
    print('src.dataIO.readSavedDatasets')

    df = pd.read_csv('data/'+file_name_df+'.csv')
    posterArray = np.load('data/'+file_name_posterArray+'.npy')

    return df, posterArray