from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from sklearn import linear_model
import numpy as np
import os as os
import glob as glob
import pandas 
from mpl_toolkits.mplot3d import Axes3D 
import functools

Chem_data = pandas.read_csv('./processed_data/Meat_meatrix_simplified.csv')
data_location = './processed_data/'
samples = []
raw_data = [700,699,695,705,703,720,692,719,708, 715, 706, 721, 698, 711, 713, 716, 707, 720, 701, 722, 705, 714, 697, 718, 693, 709, 704, 703, 712, 702, 694, 710 ]
Fat = []
Fat_lab1 = []
all_spectra = []

#for x in range(691,722,1):
for x in raw_data:
    result_spectra=[]
    for file in glob.glob(data_location + str(x) +"/grid*_Int_masked.npy"):
        spectra= np.load(file)
        result_spectra = np.concatenate((result_spectra, spectra))
        
    if len(result_spectra) != 0:
        spectra_array, edges= np.histogram(result_spectra, bins=200, range = [-0.15,0.15])
        all_spectra.append(spectra_array/max(spectra_array))
        samples.append(x)
        
        for y in range(len(Chem_data['Barcode_#'])):
            if Chem_data['Barcode_#'][y] == x:
                Fat.append(Chem_data['Fat'][y])
                Fat_lab1.append(Chem_data['Fat_lab1'][y])

#print(all_spectra)
#for i, x in enumerate(all_spectra):
#    plt.plot(x,  label = str(samples[i]))
    
#plt.legend()
#plt.show()



# instantiate the PCA class

    #spectra2 = np.array(spectra['']).reshape(len(spectra)*len(spectra['grid'][0]), len(spectra[0]['masked'][0])) # reshape the 2D fft's to fit with PCA function
pca = PCA(n_components = 5) # n_components may need to be optimised
train_transf = pca.fit_transform(all_spectra)
#print ('the explained variance by each PC: ', sum(pca.explained_variance_ratio_)) # shows the amount of variance explained by each of the selected components

var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100) #cumulative variance explains
print (var1[1])
print(var1[2])
print(var1[3])
plt.plot(var1)
plt.xlabel('Number of components or features')
plt.ylabel('% variance explained')
plt.title('Variance of different principal components')
plt.show()

loadings= pca.components_
print(loadings)


attenuation_loadings = np.linspace(-0.15,0.15, num = len(loadings[0,:]))
plt.xlabel('Attenuation')
plt.ylabel('Loadings')
plt.title('Loadings of PCA components * variance ')
loadingsPC1= (loadings[0,:])*(var1[0])
loadingsPC2= (loadings[1,:])*(var1[1]-var1[0]) 
loadingsPC3= (loadings[2,:])*(var1[2]-var1[1])  
plt.plot(attenuation_loadings, all_spectra[5], label = 'spectra')
plt.plot(attenuation_loadings, loadingsPC1, label = 'PC1')
plt.plot(attenuation_loadings, loadingsPC2, label = 'PC2')
plt.plot(attenuation_loadings, loadingsPC3, label = 'PC3')
plt.legend()
plt.grid()
plt.show()


attenuation_loadings = np.linspace(-0.15,0.15, num = len(loadings[0,:]))
plt.xlabel('Attenuation')
plt.ylabel('Loadings')
plt.title('Loadings of PCA components')
plt.plot(attenuation_loadings, all_spectra[5], label = 'spectra')
plt.plot(attenuation_loadings, loadings[0,:], label = 'PC1')
plt.plot(attenuation_loadings, loadings[1,:], label = 'PC2')
plt.plot(attenuation_loadings, loadings[2,:], label = 'PC3')
plt.legend()
plt.grid()
plt.show()



for i,x in enumerate(samples):
    print(x,Fat[i],Fat_lab1[i], 'PC1: ' + str(train_transf[i,0]), 'PC2: ' + str(train_transf[i,1]),  'PC3: ' + str(train_transf[i,2]))    

normalize = mcolors.Normalize(vmin=min(Fat), vmax=3)
normalize1= mcolors.Normalize(vmin=min(Fat_lab1), vmax=3)
colormap = cm.plasma
fig, ax = plt.subplots(2,3, figsize = (16,4))


for x in range(0,len(train_transf[:,0])):
    ax[0][0].plot(train_transf[x,0], train_transf[x,1], 'o', color =colormap(normalize(Fat[x])))
    ax[0][1].plot(train_transf[x,0], train_transf[x,2], 'o', color =colormap(normalize(Fat[x])))
    ax[0][2].plot(train_transf[x,1], train_transf[x,2], 'o', color =colormap(normalize(Fat[x])))
    ax[1][0].plot(train_transf[x,0], train_transf[x,1], 'o', color =colormap(normalize1(Fat_lab1[x])))
    ax[1][1].plot(train_transf[x,0], train_transf[x,2], 'o', color =colormap(normalize1(Fat_lab1[x])))
    ax[1][2].plot(train_transf[x,1], train_transf[x,2], 'o', color =colormap(normalize1(Fat_lab1[x])))    

#plt.title('PCA analysis with averaged fat data')
ax[0][0].title.set_text ('PCA analysis with averaged fat data')
ax[0][0].set_xlabel('PC1')
ax[0][0].set_ylabel('PC2')
ax[0][0].legend()
ax[0][0].grid()
ax[0][1].title.set_text ('PCA analysis with averaged fat data')
ax[0][1].set_xlabel('PC1')
ax[0][1].set_ylabel('PC3')
ax[0][1].legend()
ax[0][1].grid()
ax[0][2].title.set_text ('PCA analysis with averaged fat data')
ax[0][2].set_xlabel('PC2')
ax[0][2].set_ylabel('PC3')
ax[0][2].legend()
ax[0][2].grid()
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(np.array(Fat))
cbar = plt.colorbar(scalarmappaple, ax=ax[0][2])
cbar.set_label('IMF%')

ax[1][0].title.set_text ('PCA analysis with laboratory1 fat data')
ax[1][0].set_xlabel('PC1')
ax[1][0].set_ylabel('PC2')
ax[1][0].legend()
ax[1][0].grid()
ax[1][1].title.set_text ('PCA analysis with laboratory1 fat data')
ax[1][1].set_xlabel('PC1')
ax[1][1].set_ylabel('PC3')
ax[1][1].legend()
ax[1][1].grid()
ax[1][2].title.set_text ('PCA analysis with laboratory1 fat data')
ax[1][2].set_xlabel('PC2')
ax[1][2].set_ylabel('PC3')
ax[1][2].legend()
ax[1][2].grid()
plt.title('PCA analysis with lab 1 fat data')
#plt.legend()

scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(np.array(Fat_lab1))
cbar = plt.colorbar(scalarmappaple, ax=ax[1][2])
cbar.set_label('IMF%')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for x in range(0,len(train_transf[:,0])):
   threeD = ax.scatter(xs=train_transf[x,0], ys=train_transf[x,1],zs=train_transf[x,2],zdir='z', marker = 'o', c= colormap(normalize(Fat_lab1[x])), depthshade=True )
ax.set_xlabel('PC1', color='white')
ax.set_ylabel('PC2', color='white')
ax.set_zlabel('PC3', color='white')
#ax.set_facecolor('xkcd:tan')
ax.set_axis_bgcolor('black')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.zaxis.label.set_color('white')
ax.tick_params(axis='x',colors='white')
ax.tick_params(axis='y',colors='white')
ax.tick_params(axis='z',colors='white')

scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(np.array(Fat_lab1))
cbar= plt.colorbar(scalarmappaple)
#ax.legend()
plt.show()
