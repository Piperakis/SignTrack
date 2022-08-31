import zipfile
from pathlib import Path
import pandas as pd
import numpy as np
import os
from os.path import exists


def ModelPack(mpath, signs):
    '''
    Conbines the model with the feature labels 
    in one file
    '''
    filename = 'model.SignTrack'
    if exists(filename):
        try:
            os.remove(filename)
        except:
            pass
    with zipfile.ZipFile(filename, 'x') as file:
        np.save('Insights/ModelID', signs)
        file.write('Insights/ModelID.npy')
        file.write(mpath)


def ModelIDUnpack():
    '''
    Extracts the model and returns the signs the model
    is able to predict
    '''
    filename = 'model.SignTrack'
    with zipfile.ZipFile(filename, 'r') as file:
        file.extractall('tmp/')
        return "tmp/Insights/ModelID.npy"
