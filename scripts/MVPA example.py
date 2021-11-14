# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 16:26:42 2018

@author: ning

This script shows how to perform temporal decoding (trained and tested in the same
     timepoint) and temporal generalization (trained in timepoint a and tested in
     timepoint b)

"""
import os,mne

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('white')
from mne import decoding
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn import metrics,utils

def make_clf():
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(random_state = 12345,
                                           multi_class = 'ovr'))
    return clf 
if __name__ == '__main__':
    n_splits = 20
    sampling_rate = 100
    
    # load data
    epochs  = mne.read_epochs('../data/1-40 Hz/old vs new-epo.fif',preload=True)
    # downsample for speed
    epochs = epochs.resample(sampling_rate)
    saving_dir = '../results'
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    
    data = epochs.get_data() # 54 by 61 by 140 matrix
    labels = epochs.events[:,-1]#  this is [0 0 0 0 ... 0 0 1 1 1 1 ... 1 1 1]
    # temporal decoding
    decoder = decoding.SlidingEstimator(make_clf(),
                                        scoring = 'roc_auc',
                                        )
    cv = StratifiedKFold(n_splits = n_splits,
                         shuffle = True,
                         random_state = 12345,
                         )
    scores = decoding.cross_val_multiscore(decoder,
                                           X = data,
                                           y = labels,
                                           # scoring = scorer, # don't specify the scorer here
                                           cv = cv,
                                           n_jobs = -1,
                                           verbose = 1,
                                           )
    # plot the results
    fig,ax = plt.subplots(figsize = (12,8))
    ax.plot(epochs.times,
            scores.mean(0),
            linestyle = '--',
            color = 'black',
            label = 'mean',)
    se = scores.std(0) / np.sqrt(n_splits)
    ax.fill_between(epochs.times,
                    scores.mean(0) + se,
                    scores.mean(0) - se,
                    color = 'red',
                    alpha = 0.5,
                    label = 'std',)
    ax.axhline(0.5, linestyle = '-',color = 'black',alpha = 1.,)
    ax.legend(loc = 'best')
    ax.set(xlabel = 'Time (sec)',
           ylabel = 'ROC AUC',
           )
    
    # temporal generalization
    cv = StratifiedKFold(n_splits = n_splits,
                         shuffle = True,
                         random_state = 12345,
                         )
    time_gen_scores = []
    for idx_train,idx_test in cv.split(data[:,:,0],labels,):
        decoder = make_clf()
        time_gen = decoding.GeneralizingEstimator(decoder,
                                                  scoring = 'roc_auc',
                                                  n_jobs = -1,
                                                  verbose = 1,
                                                  )
        time_gen.fit(data[idx_train],labels[idx_train])
        _scores = time_gen.score(data[idx_test],labels[idx_test])
        time_gen_scores.append(_scores)
    time_gen_scores = np.array(time_gen_scores)
    time_gen_scores_mean = time_gen_scores.mean(0)
    
    # plot the results
    fig, ax = plt.subplots(figsize = (12,12))
    ax.imshow(time_gen_scores_mean,
              origin = 'lower',
              cmap = plt.cm.coolwarm,
              vmin = .4,
              vmax = .6,
              interpolation = 'hanning',
              )
    ax.set(xlabel = 'Testing time (sec)',
           ylabel = 'Training time (sec)',
           xticks = np.arange(epochs.times.shape[0])[::5],
           yticks = np.arange(epochs.times.shape[0])[::5],
           xticklabels = epochs.times[::5],
           yticklabels = epochs.times[::5],)
    
    









