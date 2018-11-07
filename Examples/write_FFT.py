import tdmsCode as pytdms
import numpy as np

data_location = '/home/sam/Meat_Raw_data/Numpy_data/'
samples = range(691,723,1)
save_location = './processed_FFT/'

for x in samples:
    save = save_location + str(x) + '/'
    read = data_location + str(x) + '/'
    pytdms.FFT_output(save, read, 5, show = False)
    
