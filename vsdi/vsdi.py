##########################################################
######################## PACKAGES ########################
##########################################################
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import norm
from sklearn.decomposition import PCA
import seaborn as sns
sns.set_theme(context='notebook',
              style='white',
              font_scale=1.5,
              rc = {'axes.spines.top':False,'axes.spines.right':False,
                    'image.cmap':plt.cm.jet})

##########################################################
####################### VSDI CLASS #######################
##########################################################

class VSDI:
    def __init__(self, vsdi, raw_mask):
        """
        Constructor for VSDI class
        """
        self.vsdi = self.correct_outliers(vsdi)
        self.raw_mask = raw_mask
        self.X = self.vsdi.transpose(2,0,1)    # reshape in time x image format
        self.T,self.h,self.w = self.X.shape    # saves time, height and width for future use
        self.X = self.X[:,self.raw_mask]       # select only cortex pixels, returns a flattened image

    def time_course(self, start_time=0, end_time=600, framerate=50):
        """
        Plot time course of a single PC
        """
        plt.figure(figsize=(10, 5))
        t = np.linspace(start_time, end_time, int((end_time - start_time) * framerate - 1))
        plt.plot(t, self.vsdi)
        plt.xlabel('Time (s)')
        plt.ylabel('PC activation (a.u.)')
        plt.show()

    def create_frame(self, t):
        """
        Create a frame of the VSDI data
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(self.vsdi[:, :, t], cmap=plt.cm.inferno)

    def pca(self, n_components=50):
        """
        Perform PCA on the VSDI data
        """
        pca = PCA(n_components=n_components)
        pca.fit(self.X)

        return pca
    
    def pca_projection(self, pca, num_pc):
        """
        Plot the projection of the data onto the first 10 PCs
        """
        PCs = pca.components_[:num_pc].T
        Y = self.X @ PCs # compute the PC timecourse, by projecting the original data on each component
        return Y

    def cumulative_explained_variance(self, pca):
        """
        Plot cumulative explained variance
        """
        evr = pca.explained_variance_ratio_
        c_evr = np.cumsum(evr)
        plt.figure(figsize=(6, 6))
        plt.axhline(0.9, linestyle='--', label='90% explained variance')
        plt.plot(range(1, len(c_evr) + 1), c_evr)
        plt.legend()
        plt.xlabel('# of components')
        plt.ylabel('EVR')

    def fingerprint(self, pca, num_pc = 10):
        """
        Plot topographic organzation of the weights of these ten components
        """
        PCs = pca.components_[:num_pc]
        plt.figure(figsize=(10, 5))
        for i, pc in enumerate(PCs):
            plt.subplot(2, 5, i + 1)
            plt.title(f'PC {i + 1}')
            reshaped_pc = np.full((self.h, self.w), np.nan)
            reshaped_pc[np.where(self.raw_mask)] = pc
            plt.imshow(reshaped_pc, aspect='auto', cmap=plt.cm.jet)
            plt.axis('off')

    def first_last_subsets(self, arr):
        """
        Get first and last frame of each subset
        """
        subsets = []
        start = 0
        end = 0
        while end < len(arr):
            while end + 1 < len(arr) and arr[end + 1] - arr[start] == end - start + 1:
                end += 1
            subsets.append((arr[start], arr[end]))
            start = end = end + 1
        return np.array(subsets)

    def correct_outliers(self, vsdi, nsigma=4):
        """
        Correct outliers in VSDI data
        """
        # Array with average value of all frames in vsdi
        mean_vsdi = np.mean(vsdi, axis=(0,1))
        std_vsdi = vsdi.std()

        # Get index of outliers from vsdi presenting average activity higher than 4 sigma
        outliers = np.argwhere((mean_vsdi > nsigma*std_vsdi) | (mean_vsdi < -nsigma*std_vsdi)).ravel()
        
        # Get first and last frame of each subset
        outliers_subsets = self.first_last_subsets(arr = outliers)

        # Set outlier frames to the mean between the previous and next frame
        for i in range(len(outliers_subsets)):
            start = outliers_subsets[i][0]
            end = outliers_subsets[i][1]
            if start == 0:
                vsdi[:,:,start:end+1] = np.tile(vsdi[:,:,end+1][:, :, np.newaxis], (1, 1, end - start + 1))
            elif end == len(mean_vsdi)-1:
                vsdi[:,:,start:end+1] = np.tile(vsdi[:,:,start-1][:, :, np.newaxis], (1, 1, end - start + 1))
            else:
                average = np.divide(np.add(vsdi[:,:,start-1][:, :, np.newaxis], vsdi[:,:,end+1][:, :, np.newaxis]), 2)
                vsdi[:,:,start:end+1] = average
        return vsdi

    def bimodality_test(self, distribution):
        """
        Test for bimodality using moving average to smooth the histogram
        and get the x-axis location of the highest point in the histogram
        """
        # Create a histogram      
        n, bins, patches = plt.hist(distribution, bins=1000)
        # Define the window size for the moving average
        window_size = 5
        # Create the moving average kernel
        kernel = np.ones(window_size) / window_size
        # Convolve the histogram data with the moving average kernel
        smoothed = np.convolve(n, kernel, mode='same')
        bins_adjusted = bins[:-1]
        
        # Find the x-axis location of the highest point in the histogram
        x_max = bins_adjusted[np.argmax(smoothed)]

        return x_max
    
    def bimodal_components(self, Y, threshold=1):
        """
        Get bimodal components
        """
        bimodal_components = []
        for i in range(Y.shape[1]):
            x_max = self.bimodality_test(Y[:,i])
            if abs(x_max) > threshold:
                bimodal_components.append([i, x_max])

        return bimodal_components