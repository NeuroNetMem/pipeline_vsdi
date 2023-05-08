import loaders
import args

import glob
import numpy as np
import pandas as pd
from scipy import io
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import vsdi_preprocessing
import vsdi_visualization
import loaders

####################################

#####################################
############### MAIN ################
#####################################

# Define paths
path = '/home/danteam/Documents/BScThesis'
data = f'{path}/data'
reports = f'{path}/reports'
figures = f'{reports}/figures'

# Get arguments from command line
parameters = args.getArgs()
subject = parameters.subject
day = parameters.day
session = parameters.session

def save_vsdi():
    vsdi1 = io.loadmat(glob.glob(f'{data}/ATC_Data_preprocessed/A0{subject}/Day1/vsdi_ATC{session}.mat')[0])['vsdi_data']
    vsdi3 = io.loadmat(glob.glob(f'{data}/ATC_Data_preprocessed/A0{subject}/Day3/vsdi_ATC{session}.mat')[0])['vsdi_data']
    vsdi5 = io.loadmat(glob.glob(f'{data}/ATC_Data_preprocessed/A0{subject}/Day5/vsdi_ATC{session}.mat')[0])['vsdi_data']
    vsdi7 = io.loadmat(glob.glob(f'{data}/ATC_Data_preprocessed/A0{subject}/Day7/vsdi_ATC{session}.mat')[0])['vsdi_data']
    
    vsdi1 = vsdi1[:,:,0:29999]
    vsdi3 = vsdi3[:,:,0:29999]
    vsdi5 = vsdi5[:,:,0:29999]
    vsdi7 = vsdi7[:,:,0:29999]
    
    vsdi = np.concatenate((vsdi1, vsdi3, vsdi5, vsdi7), axis = 2)

    del(vsdi1)
    del(vsdi3)
    del(vsdi5)
    del(vsdi7)   

    # Outliers correction
    vsdi_clean = vsdi_preprocessing.clean_outliers(vsdi, nsigma=5)

    # Hemodynamics correction
    vsdi_clean = vsdi_preprocessing.clean_hemodynamic_pca(vsdi_clean, bimodal_th = 0.6, verbose = True)

    del(vsdi)
    
    ## Mask to select just subject cortex
    r = np.loadtxt(f'{data}/Patterns/mask_A0{subject}.csv', delimiter=',', dtype='bool')

    X = vsdi_clean.transpose(2, 0, 1)
    X = X[:,r]

    del(vsdi_clean)
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=10))
    ])

    out = pipe.fit(X)

    f = out.named_steps['pca'].components_
    factor_scores = pipe.named_steps['pca'].transform(X)
    
    np.savetxt(f'{data}/Patterns/SubjectA0{subject}_timecourse_part1.csv', factor_scores, delimiter=",")
    np.savetxt(f'{data}/Patterns/SubjectA0{subject}_fingerprint_part1.csv', f, delimiter=",")

    return 0

def save_behav():
    atc1 = io.loadmat(f'{data}/ATC_Data_preprocessed/A0{subject}/Day1/ATC{session}.mat')
    atc3 = io.loadmat(f'{data}/ATC_Data_preprocessed/A0{subject}/Day3/ATC{session}.mat')
    atc5 = io.loadmat(f'{data}/ATC_Data_preprocessed/A0{subject}/Day5/ATC{session}.mat')
    atc7 = io.loadmat(f'{data}/ATC_Data_preprocessed/A0{subject}/Day7/ATC{session}.mat')
    b_data1 = loaders.extract_behavioural_data(atc1)
    b_data3 = loaders.extract_behavioural_data(atc3)
    b_data5 = loaders.extract_behavioural_data(atc5)
    b_data7 = loaders.extract_behavioural_data(atc7)

    del(atc1)
    del(atc3)
    del(atc5)
    del(atc7)

    d1 = vsdi_preprocessing.design_matrix(b_data1)
    d3 = vsdi_preprocessing.design_matrix(b_data3)
    d5 = vsdi_preprocessing.design_matrix(b_data5)
    d7 = vsdi_preprocessing.design_matrix(b_data7)

    d1 = d1[0:29999,:]
    d3 = d3[0:29999,:]
    d5 = d5[0:29999,:]
    d7 = d7[0:29999,:]

    np.savetxt(f'{data}/Patterns/SubjectA0{subject}_d1_part{session}.csv', d1, delimiter=",")
    np.savetxt(f'{data}/Patterns/SubjectA0{subject}_d3_part{session}.csv', d3, delimiter=",")
    np.savetxt(f'{data}/Patterns/SubjectA0{subject}_d5_part{session}.csv', d5, delimiter=",")
    np.savetxt(f'{data}/Patterns/SubjectA0{subject}_d7_part{session}.csv', d7, delimiter=",")

    return 0

def mask_norm(masks, threshold=0.5):
    mask_mean = np.mean(masks, axis=0)
    mask_mean[mask_mean > threshold] = 1
    mask_mean[mask_mean <= threshold] = 0

    return mask_mean

def save_mask():
    r1 = io.loadmat(f'{data}/ATC_Data_preprocessed/A0{subject}/Day1/vsdi_mask.mat')['mask'].astype('bool')
    np.savetxt(f'{data}/Patterns/raw_mask_{subject}_day1.csv', r1, delimiter=",")
    r3 = io.loadmat(f'{data}/ATC_Data_preprocessed/A0{subject}/Day3/vsdi_mask.mat')['mask'].astype('bool')
    np.savetxt(f'{data}/Patterns/raw_mask_{subject}_day3.csv', r3, delimiter=",")
    r5 = io.loadmat(f'{data}/ATC_Data_preprocessed/A0{subject}/Day5/vsdi_mask.mat')['mask'].astype('bool')
    np.savetxt(f'{data}/Patterns/raw_mask_{subject}_day5.csv', r5, delimiter=",")
    r7 = io.loadmat(f'{data}/ATC_Data_preprocessed/A0{subject}/Day7/vsdi_mask.mat')['mask'].astype('bool')
    np.savetxt(f'{data}/Patterns/raw_mask_{subject}_day7.csv', r7, delimiter=",")

    del(r1)
    del(r3)
    del(r5)
    del(r7)

    r1 = np.loadtxt(f'{data}/Patterns/raw_mask_{subject}_day1.csv', delimiter=",")
    r3 = np.loadtxt(f'{data}/Patterns/raw_mask_{subject}_day3.csv', delimiter=",")
    r5 = np.loadtxt(f'{data}/Patterns/raw_mask_{subject}_day5.csv', delimiter=",")
    r7 = np.loadtxt(f'{data}/Patterns/raw_mask_{subject}_day7.csv', delimiter=",")

    print(r1.shape, r3.shape, r5.shape, r7.shape)

    masks = np.stack([r1, r3, r5, r7])
    mask_mean = mask_norm(masks)

    np.savetxt(f'/home/danteam/Documents/BScThesis/data/Patterns/mask_A0{subject}.csv', mask_mean.astype(int), fmt='%i', delimiter=",")    

    return 0


def main():
    # print(f'Saving Mask A0{subject} part {session}')
    # save_mask()
    print(f'Saving VSDI subject A0{subject} part {session}')
    save_vsdi()
    print(f'Saving Design matrices A0{subject} part {session}')
    save_behav()

    return 0



if __name__ == '__main__':
    main()