import numpy as np
import pandas as pd

import os

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def cart2pol_3d(x):
    infinitesimal = 1e-10
    rho = np.sqrt(np.sum(np.square(x), axis=1))
    theta = np.arccos(x[:,2]/(rho+infinitesimal))
    phi = np.arctan2(x[:,1], x[:,0]+infinitesimal)
    return(rho, theta, phi)

def pol2cart_3d(rho, theta, phi):
    x = np.array(rho * np.sin(theta) * np.cos(phi)).reshape(-1,1)
    y = np.array(rho * np.sin(theta) * np.sin(phi)).reshape(-1,1)
    z = np.array(rho * np.cos(theta)).reshape(-1,1)
    res = np.append(x, np.append(y, z, axis=1), axis=1)
    return res

def get_sample_list(path, suffix):

    ct = 0
    files = []
    for file in sorted(os.listdir(path)):
        if file.endswith(suffix):
            ct += 1
            files.append(file)
    np.random.shuffle(files)
    return files

def save_excel(df, filename, sheet_name="sheet1"):
    writer = pd.ExcelWriter(filename)
    df.to_excel(writer, sheet_name)
    writer.save()
    return

