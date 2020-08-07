# Import libraries
import numpy as np
import pandas as pd
import requests
from PIL import Image


def readCSV(path, file_name, encoding):
    '''
    Simple wrapper for Pandas read_csv

    Parameters:
    path (str) - 
    file_name (str) - 
    encoding (str) - 

    Returns:
    df (pd.DataFrame) - 
    '''
    try:
        df = pd.read_csv(path+file_name, encoding=encoding)

        return df
    except:
        return 'File Not Found'


# def extractPosterArrays(df, )
