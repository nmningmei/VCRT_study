# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 18:00:19 2018

@author: asus-task
This script is to demonstrate the process of decoding old (coded 0) and new (coded 1) and scramble (coded 2) images by the ERP (EEG) signals.
1. Stack the ERPs to form the dataset
2. Split the dataset into training (80%) and testing (20%) set 
3. Split the training (64 X 61 X 1400 dimensional matrix) set with 50 ms window along the last dimension ==> (64 X 61 X 50 X 28)
4. Within each segment (along the last dimension where it is 28), a classification pipeline is trained
5. The classification pipeline contains: 
    a. vectorizer: https://martinos.org/mne/stable/generated/mne.decoding.Vectorizer.html?highlight=vectorizer
    b. standardizer: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
    c. linear SVM: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
6. These 28 classification pipelines are tested on the testing set within each segment, and performances are measure by ROC AUC
7. The training and testing set were rotated by 5-fold cross validation, thus, 28*5 = 140 ROC AUCs should be obtained
8. The order of the data is also shuffled/no shuffled to test if there is an effect of iteration order (1 no shuffle + 10 shuffle)
9. Since the testing process could happen in different time samples other than where the classification pipeline is trained, a temporal 
    generalization process is applied to obtained classification performances of a classification pipeline in which it is not trained on
10. A 5-fold cross validation is also nested with the temporal generalization
"""

############################### 3 classes ##########################
if __name__ == '__main__':
    import os
    os.chdir('D://Epochs')
    import avr_reader
    import mne
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    sns.set_style('white')
    from glob import glob
    import pickle
    os.chdir('D:/Epochs')
    from mne.decoding import LinearModel,get_coef,SlidingEstimator,cross_val_multiscore,GeneralizationAcrossTime
    from sklearn.model_selection import StratifiedKFold,permutation_test_score,cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegressionCV,SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn import metrics,utils
    from tqdm import tqdm
    from mne.decoding import Vectorizer
    from scipy import stats
    from sklearn.multiclass import OneVsOneClassifier
    epochs  = mne.read_epochs('D:/NING - spindle/VCRT_study/data/0.1-40 Hz/3 classes-epo.fif',preload=True)
    def make_clf(vec=True):
        clf = []
        if vec:
            clf.append(('vec',Vectorizer()))
        clf.append(('std',StandardScaler()))
        clf.append(('est',OneVsOneClassifier(SVC(max_iter=-1,random_state=12345,class_weight='balanced',
                                          kernel='linear',probability=False))))
        clf = Pipeline(clf)
        return clf 
    results_ = []# for saving all the results
    saving_dir = 'D:\\Epochs\\vcrt results\\'
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    ################# first iteration: not shuffling the order of the subjects #################################
    data = epochs.get_data() # 54 by 61 by 1400 matrix
    labels = epochs.events[:,-1]#  this is [0 0 0 0 ... 0 0 1 1 1 1 ... 1 1 1...2 2 2]
    results={'scores_mean':[],'scores_std':[],'clf':[],'chance_mean':[],'pval':[],'activity':[],'chance_se':[]}
    idx = np.arange(data.shape[-1]).reshape(-1,50) # 28 by 50 matrix
    
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=12345)# 5 fold cross validation
    clfs = []
    scores = []
#    patterns = []
    idx = np.arange(1400).reshape(-1,50)
    for train,test in tqdm(cv.split(data,labels),desc='train-test,no shuffle'):# split the data into training set and testing set
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
        scores_ = [metrics.f1_score(y_,clf.predict(X_[:,:,ii]),average='micro') for ii,clf in zip(idx,clfs[-1])]
        scores.append(scores_)
    scores = np.array(scores)
#    patterns=np.array(patterns)
    
    ######################### chance estimation n_perm = 10000 #############
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=12345)# 5 fold cross validation
    n_perm = 1000
    counts = 0
    chances = []
    for n_perm_ in tqdm(range(int(1e5)),desc='permutation test'):# the most outer loop of the permutation test
        try:# the stratified k fold cross validation might not work for some runs, but it doesn't matter, so I skip them
            chances_ = []# second order temporal data storage
            # during each permutation, we randomly shuffle the labels, so that there should not be any informative patterns
            # that could be learned by the classifier. In other words, the feature data does not correlate to the labels
            perm_labels = labels[np.random.choice(len(labels),size=labels.shape,replace=False)]
            for train,test in cv.split(data,labels):# do the same procedure as a real cross validation
                X = data[train]
                y = perm_labels[train]
                X_ = data[test]
                y_ = perm_labels[test]
                clfs_=[make_clf().fit(X[:,:,ii],y) for ii in idx]
                scores_ = [metrics.f1_score(y_,clf.predict(X_[:,:,ii]),average='micro') for ii,clf in zip(idx,clfs[-1])]
                chances_.append(scores_)
            chances.append(chances_)
            counts += 1
        except:
            print("something is wrong, but I don't care")
        if counts > n_perm:
            break
    chances = np.array(chances)  
    np.save(saving_dir+"chance (3 class).npy", chances)
    chances = np.load(saving_dir+"chance (3 class).npy")
    # percentage of chance scores that exceed the observed score, and if it is less than 0.05, 
    # we claim the observed score statistically significant higher than chance level
    pval = (np.array(chances.mean(1) > scores.mean(0)).sum(0)+1) / (n_perm +1) 
    
    results['scores_mean']=scores.mean(0)
    results['scores_std']=scores.std(0)
    results['chance_mean']=np.mean(chances,axis=1).mean(0)
    results['chance_se']=np.std(chances.mean(1))/np.sqrt(n_perm)# standard error
    results['clf']=clfs
    results['pval']=pval
    # average pattern learned by last dimension, which is the 50 ms window
    # average pattern learned by the classifier over 5 folds
#    results['activity']=patterns.mean(-1).mean(0)
    pickle.dump(results,open(saving_dir+'temp_no_shuffle (3 classes).p','wb'))
    results_.append(results)
    
    
    for i_random in range(10):
        data = epochs.get_data()
        labels = epochs.events[:,-1]
        results={'scores_mean':[],'scores_std':[],'clf':[],'chance_mean':[],'pval':[],'activity':[],'chance_se':[]}
        for ii in range(100):
            data,labels = utils.shuffle(data,labels)# only difference from above
        idx = np.arange(data.shape[-1]).reshape(-1,50) # 28 by 50 matrix
#        cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=12345)# 5 fold cross validation
        clfs = []
        scores = []
#        patterns = []
        idx = np.arange(1400).reshape(-1,50)
        for train,test in tqdm(cv.split(data,labels),desc='train-test, shuffle'):# split the data into training set and testing set
            X = data[train]
            y = labels[train]
            # fit a classifier at each of the 50 ms window with only the training data and record the trained classifier
            clfs.append([make_clf(True).fit(X[:,:,ii],y) for ii in idx])
            # get the decoding pattern learned by each trained classifier at each of the 50 ms window with only the training data
#            temp_patterns = np.array([get_coef(c,attr='patterns_',inverse_transform=True) for c in clfs[-1]])
#            patterns.append(temp_patterns)
            X_ = data[test]
            y_ = labels[test]
            # compute the performance of each trained classifier at each of the 50 ms window with the testing data
            scores_ = [metrics.f1_score(y_,clf.predict(X_[:,:,ii]),average='micro') for ii,clf in zip(idx,clfs[-1])]
            scores.append(scores_)
        scores = np.array(scores)
#        patterns=np.array(patterns)
            
        pval = (np.array(chances.mean(1) > scores.mean(0)).sum(0)+1) / (n_perm +1) 
    
        results['scores_mean']=scores.mean(0)
        results['scores_std']=scores.std(0)
        results['chance_mean']=np.mean(chances,axis=1).mean(0)
        results['chance_se']=np.std(chances.mean(1))/np.sqrt(n_perm)# standard error
        results['clf']=clfs
        results['pval']=pval
        # average pattern learned by last dimension, which is the 50 ms window
        # average pattern learned by the classifier over 5 folds
#        results['activity']=patterns.mean(-1).mean(0)
        pickle.dump(results,open(saving_dir+'temp_shuffle_%d (3 classes).p'%i_random,'wb'))
        results_.append(results)
    
    pickle.dump(results_,open(saving_dir+'shuffle results (old vs new).p','wb'))
    ####################################################
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=12345)
    clfs = []
    for train,test in tqdm(cv.split(data,labels),desc='training'):
        X = data[train]
        y = labels[train]
        clfs.append([make_clf().fit(X[:,:,ii],y) for ii in range(X.shape[-1])])
        
    scores_within = []
    for fold,(train,test) in tqdm(enumerate(cv.split(data,labels)),desc='test within'):
        X = data[test]
        y = labels[test]   
        scores_ = []
        for clf in clfs[fold]:
            scores_temp = [metrics.f1_score(y,clf.predict(X[:,:,ii]),average='micro') for ii in range(X.shape[-1])]
            scores_.append(scores_temp)
        scores_within.append(scores_)
    scores_within = np.array(scores_within)
    
    pickle.dump(scores_within,open(saving_dir+'temporal generalization(3 classes).p','wb'))
    scores_within = pickle.load(open(saving_dir+'temporal generalization(3 classes).p','rb'))
    
    font = {
            'weight' : 'bold',
            'size'   : 20}
    
    import matplotlib
    matplotlib.rc('font', **font)
    fig,ax = plt.subplots(figsize=(12,10))
    im = ax.imshow(scores_within.mean(0),origin='lower',aspect='auto',extent=[0,1400,0,1400],cmap=plt.cm.RdBu_r,vmin=.33)
    cbar=plt.colorbar(im)
    cbar.set_label('F1 score (micro average)')
    ax.set(xlabel='Test time',ylabel='Train time',
           title='Old vs New vs Scramble Temporal Generalization\nLinear SVM, 5-fold CV')
    fig.savefig(saving_dir+'Old vs New vs scr decoding generalization.png',dpi=500)

############### plot ######################################################################
from matplotlib import pyplot as plt
import pickle
import numpy as np
from glob import glob
working_dir = 'D:\\Epochs\\vcrt results\\'
shuffle_files = glob(working_dir+'*_shuffle*(3 classes).p')
results = [pickle.load(open(f,'rb')) for f in shuffle_files]
no_shuffle = results[0]
shuffle = results[1:] 
import mne
epochs = mne.read_epochs('D://Epochs//3 class-epo.fif',preload=False)
font = {
        'weight' : 'bold',
        'size'   : 20}
import matplotlib
matplotlib.rc('font', **font)
fig,ax = plt.subplots(figsize=(16,8))
times = np.linspace(25,1375,28)
ax.plot(times,no_shuffle['scores_mean'],color='k',alpha=1.,label='Classifi.Score (F1 Mean)_no shuffle')
m,s = np.array(no_shuffle['scores_mean']),np.array(no_shuffle['scores_std'])/np.sqrt(5)
ax.fill_between(times,m+s,m-s,color='red',alpha=.3,label='Classifi.Score (SE)')
ax.plot(times,no_shuffle['chance_mean'],color='k',linestyle='--',alpha=1.,label='Chance level (Mean)')
mm,ss = np.array(no_shuffle['chance_mean']),np.array(no_shuffle['chance_se'])
ax.fill_between(times,m+s,m-s,color='red',alpha=.7,lw=0.5)
for ii, item in enumerate(shuffle):
    if ii == 0:
        ax.plot(times,item['scores_mean'],color='blue',alpha=.7,label='Classifi.Score (F1 Mean)_shuffle')
    else:
        ax.plot(times,item['scores_mean'],color='blue',alpha=1.)
    m,s = np.array(item['scores_mean']),np.array(item['scores_std'])/np.sqrt(5)
    ax.fill_between(times,m+s,m-s,color='red',alpha=.3)
ax.set(xlabel='Time (ms)',ylabel='Classifi.Score (F1)',title='Temporal Decoding\n Old vs New vs Scramble\nLinear SVM, 5-fold, n_permutation=1000',
       xlim=(0,1400),xticks=times[::3])
pvals = np.vstack([item['pval'] for item in results[1:]])    
pvals = np.vstack((no_shuffle['pval'],pvals),)
pval_set = np.sum(pvals < 0.05, axis=0)
pval_idx = np.where(pval_set> (11/2))[0]
for ii,idx in enumerate(pval_idx):
    if ii == 0:
        ax.axvspan(times[idx]-25,times[idx]+25,color='red',alpha=.2,label='pval < 0.05')
    else:
        ax.axvspan(times[idx]-25,times[idx]+25,color='red',alpha=.2)
ax.legend(fontsize='small')   
fig.savefig('D:\\NING - spindle\\VCRT_study\\results\\'+'old vs new vs scr temporal decoding.png',dpi=500,bbox_inches = 'tight') 





















































    