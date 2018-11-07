import tdmsCode as pytdms
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import scipy.signal as sig
from mpl_toolkits.mplot3d import Axes3D
from skimage.io import imsave
import time
import glob as glob






data_location = '/media/sam/Seagate Backup Plus Drive/OCT_data/Numpy_data/'
samples = range(691,723,1)
save_location = './processed_data/'
for x in samples:

    save = save_location + str(x) + '/'
    read = data_location + str(x) + '/'
    pytdms.hist_output(save, read, 5, show = False)
    



#for x in samples:
#    save = save_location + str(x) + '/'
#    read = data_location + str(x) + '/'

#    hist_data, edges = pytdms.hist_output(save, read, 5, show = True)
