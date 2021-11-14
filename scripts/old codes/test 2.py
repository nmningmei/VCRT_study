# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 12:01:06 2018

@author: install
"""

if __name__ == '__main__':
    import os
    os.chdir('D:\\NING - spindle\\VCRT_study\\scripts')
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
    from sklearn.model_selection import StratifiedKFold,permutation_test_score,cross_val_score,GridSearchCV
    from sklearn.preprocessing import StandardScaler,QuantileTransformer,RobustScaler,Normalizer
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import Pipeline
    from sklearn import metrics,utils
    from tqdm import tqdm
    from mne.decoding import Vectorizer
    from scipy import stats

    # load the epoch data converted from BESA to MNE-python epoch object
    os.chdir('D:\\NING - spindle\\VCRT_study\\data\\0.1-40 Hz')
    epochs  = mne.read_epochs('D:/NING - spindle/VCRT_study/data/0.1-40 Hz/new vs old-epo.fif',preload=True)
    mne.combine_evoked([epochs['new'].average(),-epochs['old'].average()],'equal').plot_joint()
#    epochs.filter(0.1,40,)
    # define the classification pipeline that is used later
    def make_clf(vec=True):
        clf = []
        if vec:# if the training data has more than 2 dimensions, the vectorization must perform on the last two dimensions
            clf.append(('vec',Vectorizer()))
#        clf.append(('std',StandardScaler()))# subtract the mean and divided by the standard deviation
        clf.append(('std',StandardScaler()))
        # parameters:
        # C: penalty term. Here I choose a small penalty to obtain better classification results
        # tol: stop point criteria, if the difference between prediction and the true label is less than tol, stop
        # max_iter: set to -1 so that the classification wouldn't stop until the tol is less thatn 1e-3
        # random_state: for replication
        # class_weight: to balance the class weight in case there is any
        # kernel: linear kernel for linear classification performance
        # probability: tell the classifier to compute probabilistic predictions for the instances
#        clf.append(('est',LinearModel(SVC(C=10,max_iter=-1,random_state=12345,class_weight='balanced',
#                                          kernel='linear',probability=True,tol=0.001))))
        clf.append(('est',RandomForestClassifier(n_estimators=5,random_state=12345,class_weight='balanced')))
#        svc = SVC(C=10,max_iter=-1,random_state=12345,class_weight='balanced',
#                                          kernel='linear',probability=True,tol=0.001)
#        grid = GridSearchCV(svc,cv=4,param_grid={'C':np.logspace(-3,3,7),'tol':[0.0001,0.001,0.01]})
#        clf.append(('est',grid))
#        clf.append(('est',LinearModel(LogisticRegressionCV(Cs=np.logspace(-3,3,7),cv=3,scoring='roc_auc',
#                                                           class_weight='balanced',
#                                                           random_state=12345,
#                                                           max_iter=int(1e6)))))
        clf = Pipeline(clf)
        return clf
    data = epochs.get_data() # 54 by 61 by 1400 matrix
    labels = epochs.events[:,-1]#  this is [0 0 0 0 ... 0 0 1 1 1 1 ... 1 1 1]
    for ii in range(100):
            data,labels = utils.shuffle(data,labels)# the only difference from above
    results={'scores_mean':[],'scores_std':[],'clf':[],'chance_mean':[],'pval':[],'activity':[],'chance_se':[]}
    interval_ = 50
    idx = np.arange(data.shape[-1]).reshape(-1,interval_) # 28 by 50 matrix, and this is for indexing the training and testing data to select the segments
    
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=12345)# 5 fold stratified cross validation
    clfs = []
    scores = []
    patterns = []
#    fig, axes = plt.subplots(figsize=(25,16),nrows=4,ncols=7)
    times = np.vstack([np.arange(0,1400+interval_,interval_)[:-1],np.arange(0,1400+interval_,interval_)[1:]]).T 
    for train,test in tqdm(cv.split(data,labels),desc='train-test'):# split the data into training set and testing set
        X = data[train]# 44 by 61 by 1400
        y = labels[train]# 44 labels
        # fit a classifier at each of the 50 ms window with only the training data and record the trained classifier
        clfs.append([make_clf(True).fit(X[:,:,ii],y) for ii in idx])
        # get the decoding pattern learned by each trained classifier at each of the 50 ms window with only the training data
#        temp_patterns = np.array([get_coef(c,attr='patterns_',inverse_transform=True) for c in clfs[-1]])
#        patterns.append(temp_patterns)
        X_ = data[test]
        y_ = labels[test]
        # compute the performance of each trained classifier at each of the 50 ms window with the testing data
        scores_ = [metrics.roc_auc_score(y_,clf.predict_proba(X_[:,:,ii])[:,-1]) for ii,clf in zip(idx,clfs[-1])]
        scores.append(scores_)
#        # plot roc curves of the decoding and save them
#        rocs = np.array([metrics.roc_curve(y_,clf.predict_proba(X_[:,:,ii])[:,-1]) for ii,clf in zip(idx,clfs[-1])])
#        
#        for ii,(roc_,ax,(start,stop)) in enumerate(zip(rocs,axes.flatten(),times)):
#            fpr,tpr,th = roc_
#            ax.plot(fpr,tpr,color='blue',)
#            ax.set(xlim=(0,1),ylim=(0,1),)
#            ax.plot([0, 1], [0, 1], linestyle='--',color='red')
#            ax.set(title='%d-%d ms'%(start,stop))
#            
#            
    scores = np.array(scores)
#    patterns=np.array(patterns)
    
    fig,ax = plt.subplots(figsize=(16,8))
    ax.plot(times.mean(1),scores.mean(0))
    ax.axhline(0.5,)
    ax.fill_between(times.mean(1),scores.mean(0)+scores.std(0),scores.mean(0)-scores.std(0),alpha=.3,color='r')
