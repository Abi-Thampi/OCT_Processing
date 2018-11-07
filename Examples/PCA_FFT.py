from sklearn.decomposition import PCA
import tdmsCode as pytdms
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from sklearn import linear_model
import functools
import numpy as np
import os as os
import glob as glob
import pandas 

Chem_data = pandas.read_csv('./processed_data/Meat_meatrix_simplified.csv')
data_location = './processed_FFT/'

samples = []
Fat = []
all_spectra = []

for x in range(691,722,1):
#for x in raw_data:
    result_spectra=[]
    for file in glob.glob(data_location + str(x) +"/grid*_FFT.npy"):
        spectra= np.load(file)
        result_spectra.append(spectra)

    if len(result_spectra) != 0 :
        spectra_array = np.mean(result_spectra, axis = 0)

        all_spectra.append(spectra_array/max(spectra_array))
        samples.append(x)
        for y in range(len(Chem_data['Barcode_#'])):
            if Chem_data['Barcode_#'][y] == x:
                Fat.append(Chem_data['avg_SF_(kPa)'][y])



#for i, x in enumerate(all_spectra):
#    plt.plot(x,  label = str(samples[i]))
    
#plt.legend()
#plt.show()
# instantiate the PCA class
#spectra2 = np.array(spectra['']).reshape(len(spectra)*len(spectra['grid'][0]), len(spectra[0]['masked'][0])) # reshape the 2D fft's to fit with PCA function

pca = PCA(n_components = 30) # n_components may need to be optimised
train_transf = np.around(pca.fit_transform(all_spectra), 3)

#print ('the explained variance by each PC: ', sum(pca.explained_variance_ratio_)) # shows the amount of variance explained by each of the selected components

var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100) #cumulative variance explains
#plt.plot(var1)
#plt.xlabel('Number of components or features')
#plt.ylabel('% variance explained')
#plt.title('Variance of different principal components')
#plt.show()
for i,x in enumerate(samples):
    print(x,Fat[i], 'PC1: ' + str(train_transf[i,0]), 'PC2: ' + str(train_transf[i,1]),  'PC3: ' + str(train_transf[i,2]))    

normalize = mcolors.Normalize(vmin=max(Fat), vmax=max(Fat))
colormap = cm.plasma # pylint: disable = E1101

fig,ax = plt.subplots(1,3)

ax[0].scatter(train_transf[:,0], train_transf[:,1], c = Fat, cmap = colormap, s = 30)
ax[1].scatter(train_transf[:,0], train_transf[:,2],  c = Fat, cmap = colormap, s = 30)
ax[2].scatter(train_transf[:,1], train_transf[:,2],  c = Fat, cmap = colormap, s = 30)
    
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[0].legend()
ax[1].set_xlabel('PC1')
ax[1].set_ylabel('PC3')
ax[1].legend()
ax[2].set_xlabel('PC2')
ax[2].set_ylabel('PC3')
ax[2].legend()

fig.canvas.mpl_connect("motion_notify_event", functools.partial(pytdms.hover,samples))

scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(np.array(Fat))
cbar = plt.colorbar(scalarmappaple)
cbar.set_label('Shear Force')

plt.show()