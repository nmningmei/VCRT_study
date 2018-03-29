# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 16:26:42 2018

@author: ning
"""

if __name__ == '__main__':
    import os
#    os.chdir('D://Epochs')
    import avr_reader 
    import mne
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    sns.set_style('white')
    from glob import glob
    import pickle
    from mne.decoding import LinearModel,get_coef,SlidingEstimator,cross_val_multiscore,GeneralizationAcrossTime
    from sklearn.model_selection import StratifiedKFold,permutation_test_score,cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn import metrics,utils
    from tqdm import tqdm
    from mne.decoding import Vectorizer
    from scipy import stats
    os.chdir('C:\\Users\\ning\\OneDrive\\python works\\VCRT_study\\data')
    epochs  = mne.read_epochs('old vs new-epo.fif',preload=True)
    def make_clf(vec=False):
        clf = []
        if vec:# if the training data has more than 2 dimensions, the vectorization must perform on the last two dimensions
            clf.append(('vec',Vectorizer()))
        clf.append(('std',StandardScaler()))# subtract the mean and divided by the standard deviation
        # parameters:
        # max_iter: set to -1 so that the classification wouldn't stop until the tol is less thatn 1e-3
        # random_state: for replication
        # class_weight: to balance the class weight in case there is any
        # kernel: linear kernel for linear classification performance
        # probability: tell the classifier to compute probabilistic predictions for the instances
        clf.append(('est',LinearModel(SVC(max_iter=-1,random_state=12345,class_weight='balanced',
                                          kernel='linear',probability=True))))
        clf = Pipeline(clf)
        return clf 
    results_ = []# for saving all the results
    saving_dir = 'C:\\Users\\ning\\OneDrive\\python works\\VCRT_study\\results\\'
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    ################# first iteration: not shuffling the order of the subjects #################################
    data = epochs.get_data() # 54 by 61 by 1400 matrix
    labels = epochs.events[:,-1]#  this is [0 0 0 0 ... 0 0 1 1 1 1 ... 1 1 1]
    data,labels = utils.shuffle(data,labels)
    results={'scores_mean':[],'scores_std':[],'clf':[],'chance_mean':[],'pval':[],'activity':[],'chance_se':[]}
    idx = np.arange(data.shape[-1]).reshape(-1,50)[4:7] # 28 by 50 matrix, and this is for indexing the training and testing data to select the segments
    
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=12345)# 5 fold stratified cross validation
    clfs = []
#    fig, axes = plt.subplots(figsize=(5,16),nrows=3,ncols=1)
    colors = ['k','blue','yellow','green','red']
    times = np.vstack([np.arange(0,1450,50)[:-1],np.arange(0,1450,50)[1:]]).T 
    for (train,test),color in zip(cv.split(data,labels),colors):# split the data into training set and testing set
        X = data[train]
        y = labels[train]
        # fit a classifier at each of the 50 ms window with only the training data and record the trained classifier
        clfs.append([make_clf(True).fit(X[:,:,ii],y) for ii in idx])
        # get the decoding pattern learned by each trained classifier at each of the 50 ms window with only the training data
#        temp_patterns = np.array([get_coef(c,attr='patterns_',inverse_transform=True) for c in clfs[-1]])
#        patterns.append(temp_patterns)
        X_ = data[test]
        y_ = labels[test]
        # compute the performance of each trained classifier at each of the 50 ms window with the testing data
        scores_ = [metrics.roc_auc_score(y_,clf.predict_proba(X_[:,:,ii])[:,-1]) for ii,clf in zip(idx,clfs[-1])]
#        scores.append(scores_)
        # plot roc curves of the decoding and save them
        rocs = np.array([metrics.roc_curve(y_,clf.predict_proba(X_[:,:,ii])[:,-1]) for ii,clf in zip(idx,clfs[-1])])
        for clf,time,s in zip(clfs[-1],times[4:7],scores_):
            c = clf.steps[-1][-1].model
            print(  'time:',time,'\n',
                    'train:',X.shape[0],'test:',X_.shape[0],'\n',
                    'score:',s,'support:',c.support_.shape[0],
                    '\n')


        for ii,(roc_,ax,(start,stop)) in enumerate(zip(rocs,axes.flatten(),times[4:7])):
            fpr,tpr,th = roc_
            ax.plot(fpr,tpr,color=color,)
            ax.set(xlim=(0,1),ylim=(0,1),)
            ax.plot([0, 1], [0, 1], linestyle='--',color='red')
            ax.set(title='%d-%d ms'%(start,stop))
    