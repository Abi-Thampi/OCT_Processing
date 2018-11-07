from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from sklearn import linear_model
import numpy as np
import os as os
import glob as glob
import pandas 

Chem_data = pandas.read_csv('./processed_data/Meat_meatrix_simplified.csv')
data_location = './processed_FFT/'
samples = []
#raw_data = [700,699,695,710,694,722,692,719,708]
Fat = []
all_spectra = []

for x in range(691,722,1):
#for x in raw_data:
    result_spectra=[]
    for file in glob.glob(data_location + str(x) +"/grid*_FFT.npy"):
        spectra= np.load(file)
        result_spectra.append(spectra)

    if len(result_spectra) != 0:
        spectra_array = np.mean(result_spectra, axis = 0)

        all_spectra.append(np.argmax(spectra_array[20:]))
        samples.append(x)
        for y in range(len(Chem_data['Barcode_#'])):
            if Chem_data['Barcode_#'][y] == x:
                Fat.append(Chem_data['avg_SF_(kPa)'][y])


plt.plot(Fat, all_spectra, 'x')
plt.show()