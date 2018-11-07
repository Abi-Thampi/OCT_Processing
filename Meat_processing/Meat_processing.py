from nptdms import TdmsFile,TdmsObject,TdmsWriter,ChannelObject
import matplotlib.pylab as plt
from matplotlib.collections import PathCollection
import numpy as np
from PIL import Image
from scipy import signal,ndimage
import numpy.ma as ma
import os as os
import glob as glob
import struct


def depth_detect(Mat, surface, threshold, length = 10):
    ## This function operates similar to surface_detect. If all the next n = length values are below the threshold then record the surface location
    depth = np.zeros_like(surface, dtype =int)
    for i in range(0,len(Mat[0,:])):
        count = 0
        for j in range(surface[i],len(Mat[:,i]-length)):
            if Mat[j,i] < threshold:
                count += 1
            else:
                count = 0
		
            if count == length:
                depth[i] = j - length
                break
        else:
            depth[i] = len(Mat[:,i] - length)

    return(depth)


def surface_detect(Mat, threshold, length = 10, skip = 0):

# This function detects the surface location of a sample in an OCT image. Inputs are the B-scan Matrix (Mat), threshold is the value used to find the surface and length is used to minimize triggering on false surfaces.
    surface = np.zeros_like(Mat[0,:], dtype = int)
    for i in range(0,len(Mat[0,:])): #Index through columns
        count = 0
        for j in range(0,len(Mat[:,i])- length): # Index through intensity values
            if Mat[j,i] > threshold:
                count+=1
            else:
                count = 0
            if count == length:
                surface[i] = j + skip
                break
        else:
            surface[i] = len(Mat[:,i])- length
    return(surface)

def atten_Int(Intmean, Depth,voxA, voxB, jumpA):
    len_a = range(0,len(Intmean[0,:,0])- voxA, jumpA)
    len_b = range(0,len(Intmean[0,0,:]), voxB)
    masked_c = np.zeros((np.shape(Intmean)[0],len(len_a),len(len_b)))
    atten_c = np.zeros_like(masked_c)
    for x in range(0,np.shape(Intmean)[0]):
        for k,z in enumerate(len_b):
            end = np.mean(Depth[x,z:z+voxB]).astype(int)
            vox = np.mean(Intmean[x,:,z:z+voxB], axis = 1)
            logmeanA = np.log(vox)
            logmeanA[np.argwhere(np.isnan(logmeanA))] = logmeanA[np.argwhere(np.isnan(logmeanA)) - 1]

            for j,y in enumerate(len_a):
                #print(y,j)
                p = np.polyfit(np.arange(0,len(logmeanA[y:y+voxA])), logmeanA[y:y+voxA], 1)
                atten_c[x,j,k] = p[0]
                if y+voxA < end:
                    pass
                else:
                    masked_c[x,j,k] = 1

    return(atten_c, masked_c)


def birefringence(Ret, start, end, wavelength = LAMBDA_CONSTANT, nref = NREF):
    #Calculate the birefringence of an A-Scan 

    ### Select part of the image for analysis
    RetSection = Ret[start:end]

    ### Smooth the image a bit
    smoothRet = ndimage.filters.gaussian_filter(RetSection,3,0)

    ### calculate the depth of the image
    depthpix = (end-start)*(10/nref)*(10**(-6))

    ### calculate the cumulative change in phase
    CumulativePhase = np.cumsum(np.absolute(np.diff(smoothRet)))*wavelength/(depthpix)
    return(CumulativePhase)


def AttCoeff(A_scan,start,end):
    p = np.polyfit(np.arange(start,end,1), A_scan[start:end], 1)
    return(p)    

def Voxel_Ret(vox):
    meanA = np.mean(np.mean(vox, axis = 0), axis = 1)
    phase = birefringence(meanA,0,len(meanA))
    biref = phase[-1]        
    return(biref)

def Heatmap_Int(Int, voxC=5, voxB=15, voxA = 20):
    jumpA = 5
    jumpB = 2
    jumpC = 2
    #set up arrays
    atten_b =[]
    atten_a = []
    atten_c = []
    Intmean = []
    Depth = []
  # first average the B-scans 
    for x in range(0,np.shape(Int)[0], voxC):
        mean = np.mean((Int[x:x+voxC,:,:]),axis = 0)
        threshold = np.mean(mean)
        surface = np.zeros(len(mean[0,:])).astype(int)
        depth1 = depth_detect(mean,surface,threshold/2, length=5)
        depth = signal.savgol_filter(depth1, 51,2).astype(int)
        #plt.figure()
        #plt.imshow(mean,cmap = 'binary')
        #plt.plot(surface, 'r')
        #plt.plot(depth,'g')
        #plt.clim(6000,20000)
        #plt.colorbar()
        #plt.show()
        Depth.append(depth)
        Intmean.append(mean)

    Intmean = np.array(Intmean)
    Depth = np.array(Depth)
    # for each averaged B-scan divide the image into voxels and calculate the attenuation

    atten_c, masked_c = atten_Int(Intmean, Depth, voxA, voxB,jumpA)
    return(atten_c, masked_c, Intmean)

def Roll_image(Im, threshold, length = 5, skip = 10):
    ### this function 'rolls' the input image such that the first index of each A_scan is the first detection of the sample. 
    ### The image is rolled in-place, and the output is an array listing the surface location of the sample BEFORE the rolling have been applied.
    ### threshold and length are inputs to find the surface, and Im is the 3D input matrix (C_Scan).

    Surface = []

    im_zeros = np.zeros_like(Im)

    for i,x in enumerate(Im):
        ### find the surface of each B-scan using surface_detect.
        surface1 =  surface_detect(x, threshold,length = 5, skip = 10)
        ### smooth the surface in case of bumps in the image.
        surface = signal.savgol_filter(surface1, 51,2).astype(int)
        ### append the surface of the B-Scan to C-Scan surface matrix.
        Surface.append(surface)
        ### Roll the image to the surface
        for y in range(len(surface)):
            data = x[surface[y]:,y]
            im_zeros[i,0:len(data),y] =  data
    ### update the input matrix with the rolled image.  
    Im[:,:,:] = im_zeros
    ### Return the surface array.
    return (np.array(Surface))
    

def hist_output(save_location, data_location, B_averages, show = True):
    ### This function creates a 1D array with attenuation values calculated from intensity C_scans.
    ### The B-scans within each C-scan are Rolled and masked so that data above the surface of the sample and data that is too deep to get good SNR are removed.

    # if the save location doesn't exit, make the directory
    if not os.path.exists(save_location):
                os.makedirs(save_location)

    ### for each grid in the data location run the processing code.
    for file in glob.glob(data_location + "grid*_Int.npy"):
        # read the gridname from the file name
        gridname = os.path.splitext(file)[0].replace(data_location, '')
        # keep a log of what grids have been written.
        print (save_location + gridname)

        # Check if the saved data already exists. if it doesn't, run the code.
        if not os.path.isfile(save_location + gridname + '_masked.npy'):

            # load the data and do some trimming
            Int = np.load(file)
            Int = np.array(Int[0:150,0:500,0:700])

            # apply the Roll function and find the surface. 
            Surface = Roll_image(Int,threshold = np.mean(Int))

            ### define voxel size
            voxC = B_averages # number of B-scans to average
            voxA =int(40*voxC/10) # size of A-scan 
            voxB =int(40*voxC/14) # size of B_scan segments

            ### calculate the attenuation for each voxel and find the mask using the Heatmap_int function
            atten_c, mask, Intmean = Heatmap_Int(Int,voxC = voxC, voxB = voxB, voxA = voxA)
            
            ### Create a 1D array of attenuation values using the mask and attenuation B_scans 
            masked_c = ma.array(atten_c, mask = mask)
            masked_c = ma.compressed(masked_c)
            ### remove any bad values
            masked_c[np.isnan(masked_c)] = 0
            ### save the data in the save location.
            np.save(save_location + gridname + '_masked.npy', masked_c)
            if show == True:
                plt.figure()
                plt.hist(masked_c, bins = 100)
                plt.xlabel('Attenuation[cm-1]')
                plt.ylabel('Number of voxels')
                plt.show()
        

def FFT_output(save_location, data_location, B_averages, show = True):

    if not os.path.exists(save_location):
                os.makedirs(save_location)
    for file in glob.glob(data_location + "grid*_Ret.npy"):
        gridname = os.path.splitext(file)[0].replace(data_location, '')
        print (save_location + gridname)
        if os.path.isfile(save_location + gridname + '_FFT.npy'):

            Ret = np.load(file)
            Ret = np.array(Ret[0:150,0:500,0:700])
            FFT = []
        ### define voxel size
            
            voxC = B_averages # number of B-scans to average

            ylen = np.shape(Ret[0,:,:])[1]
            xlen = np.shape(Ret[0,:,:])[0]

            padding = np.zeros((2048, ylen)) 
            window = np.hanning(xlen)
            for x in range(0,np.shape(Ret)[0], voxC):
                mean = np.mean((Ret[x:x+voxC,:,:]),axis = 0)   - np.pi/4  
                mean = (mean.T * window).T
                padding[int(1024 - (xlen/2)): int(1024 + (xlen/2)),:] = mean
               
                avg_FFT_B_scan = np.mean(abs(np.fft.fft(padding, axis = 0)), axis = 1)[0:1024]
                FFT.append(avg_FFT_B_scan)
                
            np.save(save_location + gridname + '_FFT.npy', np.mean(FFT, axis = 0))
        #hist, edges = np.histogram(masked_c, bins = 100, density=False)
        #np.save(save_location + gridname + '_hist.npy', np.array([hist, edges]))




annot = None
def hover( annotations, event):

    ### Example use:
#import tdmsCode as pytdms
#import matplotlib.pyplot as plt
#import numpy as np
#import functools

#annotations = ['Fred', 'Sam', 'Abi', 'Matt', 'Sylwia', 'Cushla', 'Mykola', 'Magda', 'Rachel', 'James']
#weights = np.random.randint(45, 100, len(samples))
#heights = np.random.randint(150, 190, len(samples))
#fig,ax = plt.subplots()
#ax.scatter(heights, weights)
#ax.set_ylabel('Height (cm)')
#ax.set_xlabel('Weight (kg)')
#fig.canvas.mpl_connect("motion_notify_event", functools.partial(pytdms.hover,annotations)) #### this is where you call the hover code
#plt.show()

    global annot
    ax = event.inaxes
    if ax is None:
        return

    ### here sc is the data points collected from the figure
    sc = ax.collections[-1]
    fig = ax.get_figure()
    if annot is not None:
        annot.set_visible(False)

    ### first make an annotation without text and hide it (set_visible(False)) 
    annot = ax.annotate("", xy=(0,0), xytext=(0,0),textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    vis = annot.get_visible()
    
    ### In this case an event occurs when the cursor is moving on the figure
    if event.inaxes == ax:
        ### check if a cursor location is incident on a data point. cont is a boolean to make the code run and ind is the index (co-ordinates of the cursor on the figure)
        cont, ind = sc.contains(event)
        
        ### if cont == True then your mouse is on a data point, then the code will run the annotation update code.
        if cont:

            ### set the location of the annotation to be the cursor location    
            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
    

            ### update the text of the annotation using the annotations given by hover
            text = "ID: {}".format(" ".join([str(annotations[n]) for n in ind["ind"]]))
            annot.set_text(text)
            ### set the annotation to visible
            annot.set_visible(True)
            ### draw the new figure
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()