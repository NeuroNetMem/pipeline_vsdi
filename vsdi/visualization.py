##########################################################
######################## PACKAGES ########################
##########################################################
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import sem
from scipy.ndimage import gaussian_filter1d
sns.set_theme(context='notebook',
              style='white',
              font_scale=1.5,
              rc = {'axes.spines.top':False,'axes.spines.right':False,
                    'image.cmap':plt.cm.jet})

##########################################################

def plot_timecourse_histograms(timecourses, nbins_rule = 'freedman-diaconis'):
    """
    Plot the histogram of the timecourses of the first 10 PCs
    
    Parameters
    ----------
    timecourses : numpy.ndarray (n_timepoints, n_components)
        Timecourses of the first 10 PCs
    num_bins : int
        Number of bins for the histogram
    """
    plt.figure(figsize=(15, 25))

    for i, tc in enumerate(timecourses.T):
        # Freedman-Diaconis rulex
        if nbins_rule == 'freedman-diaconis':
            iqr = np.quantile(tc, 0.75) - np.quantile(tc, 0.25)
            bin_width = (2 * iqr) / (len(tc) ** (1 / 3))
            nbins = int(np.ceil((tc.max() - tc.min()) / bin_width)) 

        # Plot histogram
        plt.subplot(5, 2, i+1)
        plt.title(f'PC {i+1}')
        plt.hist(tc, bins=nbins)
        plt.xlim(-2*np.std(tc), 2*np.std(tc))

def plot_timecourses(timecourses):
    """
    Plot the timecourses of the first 10 PCs

    Parameters
    ----------
    timecourses : numpy.ndarray (n_timepoints, n_components)
        Timecourses of the first 10 PCs
    """
    start_time = 0
    end_time = 600
    # framerate = 50
    t = np.linspace(start_time,end_time,timecourses.shape[0])

    plt.figure(figsize=(15, 25))
    for i, tc in enumerate(timecourses.T):
        plt.subplot(5, 2, i+1)
        plt.plot(t, tc)
        plt.title(f'PC {i+1}')

def plot_fingerprints(PCs, raw_mask):
    """
    Plot the topographic organization of the weights of the first 10 PCs
    
    Parameters
    ----------
    PCs : numpy ndarray
        PCs matrix (n_components, n_pixels)
    raw_mask : ndarray
        Mask of the brain cortex
    """
    plt.figure(figsize=(10,5))
    for i, pc in enumerate(PCs):
        plt.subplot(2,5,i+1)
        plt.title(f'PC {i+1}')
        reshaped_pc = np.full((raw_mask.shape[0], raw_mask.shape[1]), np.nan)
        reshaped_pc[np.where(raw_mask)] = pc
        plt.imshow(reshaped_pc, aspect='auto', cmap=plt.cm.jet)
        plt.axis('off')

def create_frame(vsdi, t):
    # VSDI - Should be already masked
    plt.imshow(vsdi[:,:,t], cmap=plt.cm.jet)
    plt.axis('off')

def fingerprint_gif(vsdi, to, tf, step):
    plt.figure(figsize=(20,15))
    for i, t in enumerate(range(to, tf, step)):
        plt.subplot(1, step, i+1)
        create_frame(vsdi, t)

def plot_betas(results):
    # Get the beta coefficients and their names
    coefficients = results.params
    names = ['Intercept'] + ["CS+ Onset", "CS+", "CS- Onset", "CS-", "Reward", "Lick"]
    colors = ['#3C3C3C', '#1E70AA', '#8BBEE2', '#C11910', '#FFB3AF', 'green','#832F95']

    # Plot the beta coefficients
    plt.figure(figsize=(2, 3))
    plt.rc('xtick',labelsize=9)
    plt.rc('ytick',labelsize=9)
    plt.bar(names, coefficients, color=colors)
    plt.xticks(ticks=range(len(names)), labels=names, rotation=-90)
    # plt.show()

def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return ""

def timecourse_around(t, d, to, tf, feature, comp):
    """
    Plot the timecourse of a given feature around a given timepoint
    
    Parameters
    ----------
    t : numpy.ndarray (n_components, n_timepoints)
        Timecourses of the first 10 PCs
    d : numpy.ndarray (n_timepoints, n_features)
        Design matrix
    to : int
        Start timepoint of interest
    tf : int
        Final timepoint
    feature : int
        Feature of interest in the following order:
        ["CS+ Onset", "CS+", "CS- Onset", "CS-", "Reward", "Lick"]
    comp : int
        Principal component to plot
    """

    names = ["CS+", "CS+ Onset", "CS-", "CS- Onset", "Reward", "Lick"]
    colors = ['#1E70AA', '#8BBEE2', '#C11910', '#FFB3AF', 'green', '#832F95']
    frame_rate = 50
    start_time = to
    end_time = tf
    peri_Y = [] # empty list for peri-lick timecourses

    for i, j in enumerate(d[:,feature]):
        try:
            if (j == 1) and d[(i-1),feature] == 0:
                peri_Y.append(t[:,(i+(frame_rate*start_time)):(i+(frame_rate*end_time))]) # saves the slice of Y 
        except:
            pass

    peri_Y = np.asarray(peri_Y)
    mean_Y = np.mean(peri_Y, axis=0) # average over licks
    error = sem(peri_Y, axis=0) # compute sem over licks

    y = mean_Y[comp-1,:]
    err = error[comp-1,:]
    
    y = gaussian_filter1d(y,2)
    err = gaussian_filter1d(err,2)

    t = np.linspace(start_time, end_time, int((end_time-start_time)*frame_rate))
    
    plt.plot(t, y, color = colors[feature])
    plt.fill_between(t, y-err, y+err, alpha=0.2, color=colors[feature], cmap="Blues")
    plt.xticks(np.arange(start_time, end_time+1, 1.0))
    plt.axvline(x=0,linestyle='--', color = "black", label=f'{names[feature]}')
    plt.legend(prop={'size': 6})
