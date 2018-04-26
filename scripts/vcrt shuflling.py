# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 17:52:29 2017

@author: ning


This script is to demonstrate the process of decoding old (coded 0) and new (coded 1) images by the ERP (EEG) signals.
1. Stack the ERPs to form the dataset
2. Split the dataset into training (80%) and testing (20%) set 
3. Split the training (43 X 61 X 1400 dimensional matrix) set with 50 ms window along the last dimension ==> (43 X 61 X 50 X 28)
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
    from sklearn.model_selection import StratifiedKFold,permutation_test_score,cross_val_score
    from sklearn.preprocessing import StandardScaler,Normalizer
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn import metrics,utils
    from tqdm import tqdm
    from mne.decoding import Vectorizer
    from scipy import stats

    # load the epoch data converted from BESA to MNE-python epoch object
    os.chdir('D:\\NING - spindle\\VCRT_study\\data\\0.1-40 Hz')
    epochs  = mne.read_epochs('D:/NING - spindle/VCRT_study/data/0.1-40 Hz/new vs old-epo.fif',preload=True)
#    epochs.filter(0.1,40,)
    # define the classification pipeline that is used later
    def make_clf(vec=True):
        clf = []
        if vec:# if the training data has more than 2 dimensions, the vectorization must perform on the last two dimensions
            clf.append(('vec',Vectorizer()))
#        clf.append(('std',StandardScaler()))# subtract the mean and divided by the standard deviation
        clf.append(('std',Normalizer()))
        # parameters:
        # C: penalty term. Here I choose a small penalty to obtain better classification results
        # max_iter: set to -1 so that the classification wouldn't stop until the tol is less thatn 1e-3
        # random_state: for replication
        # class_weight: to balance the class weight in case there is any
        # kernel: linear kernel for linear classification performance
        # probability: tell the classifier to compute probabilistic predictions for the instances
        clf.append(('est',LinearModel(SVC(C=100,max_iter=-1,random_state=12345,class_weight='balanced',
                                          kernel='linear',probability=True,tol=0.001))))
        clf = Pipeline(clf)
        return clf 
    results_ = []# for saving all the results
    saving_dir = 'D:\\NING - spindle\\VCRT_study\\results\\'
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)
    ################# first iteration: not shuffling the order of the subjects #################################
    data = epochs.get_data() # 54 by 61 by 1400 matrix
    labels = epochs.events[:,-1]#  this is [0 0 0 0 ... 0 0 1 1 1 1 ... 1 1 1]
    results={'scores_mean':[],'scores_std':[],'clf':[],'chance_mean':[],'pval':[],'activity':[],'chance_se':[]}
    idx = np.arange(data.shape[-1]).reshape(-1,50) # 28 by 50 matrix, and this is for indexing the training and testing data to select the segments
    
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=12345)# 5 fold stratified cross validation
    clfs = []
    scores = []
    patterns = []
#    fig, axes = plt.subplots(figsize=(25,16),nrows=4,ncols=7)
    times = np.vstack([np.arange(0,1450,50)[:-1],np.arange(0,1450,50)[1:]]).T 
    for train,test in tqdm(cv.split(data,labels),desc='train-test'):# split the data into training set and testing set
        X = data[train]# 44 by 61 by 1400
        y = labels[train]# 44 labels
        # fit a classifier at each of the 50 ms window with only the training data and record the trained classifier
        clfs.append([make_clf(True).fit(X[:,:,ii],y) for ii in idx])
        
        # get the decoding pattern learned by each trained classifier at each of the 50 ms window with only the training data
        temp_patterns = np.array([get_coef(c,attr='patterns_',inverse_transform=True) for c in clfs[-1]])
        if temp_patterns.shape[-1] != 50:
            vecl = Vectorizer().fit(X[:,:,idx[0]])
            temp_patterns = np.array([vecl.inverse_transform(temp.reshape(1,-1))[0] for temp in temp_patterns])
        patterns.append(temp_patterns)
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
    patterns=np.array(patterns)
    
    ##############################################################################################
    ##############################################################################################
    ######################### chance estimation n_perm = 10000 #############
    ###### to get the chance level performance, you can uncomment level 122 to line 144 #########
    ###### It it going to take very long time #############
    ###### 1. randomly shuffle the labels while the order of the feature data matrix remain the same
    ###### 2. Perform the same cross validation as shown above
    ###### 3. performances are obtained by ROC AUC
    
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=12345)# 5 fold cross validation
    n_perm = 1000
    counts = 0
    chances = []
#    for n_perm_ in tqdm(range(int(1e5)),desc='permutation test'):# the most outer loop of the permutation test
#        try:# the stratified k fold cross validation might not work for some runs, but it doesn't matter, so I skip them
#            chances_ = []# second order temporal data storage
#            # during each permutation, we randomly shuffle the labels, so that there should not be any informative patterns
#            # that could be learned by the classifier. In other words, the feature data does not correlate to the labels
#            perm_labels = labels[np.random.choice(len(labels),size=labels.shape,replace=False)]
#            for train,test in cv.split(data,labels):# do the same procedure as a real cross validation
#                X = data[train]
#                y = perm_labels[train]
#                X_ = data[test]
#                y_ = perm_labels[test]
#                clfs_=[make_clf().fit(X[:,:,ii],y) for ii in idx]
#                scores_ = [metrics.roc_auc_score(y_,clf.predict_proba(X_[:,:,ii])[:,-1]) for ii,clf in zip(idx,clfs_)]
#                chances_.append(scores_)
#            chances.append(chances_)
#            counts += 1
#        except:
#            print("something is wrong, but I don't care")
#        if counts > n_perm:
#            break
#        
#    chances = np.array(chances)  
#    np.save(saving_dir+"chance (old vs new).npy",chances)
    chances = np.load(saving_dir+"chance (old vs new).npy")
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
    results['activity']=patterns.mean(-1).mean(0)
    pickle.dump(results,open(saving_dir+'temp_no_shuffle (old vs new).p','wb'))
    results_.append(results)

    
    for i_random in range(10):# experimenting how shuffling the data affects the classification scores
        data = epochs.get_data() # 54 by 61 by 1400 matrix
        labels = epochs.events[:,-1]#  this is [0 0 0 0 ... 0 0 1 1 1 1 ... 1 1 1]
        results={'scores_mean':[],'scores_std':[],'clf':[],'chance_mean':[],'pval':[],'activity':[],'chance_se':[]}
        #### within this for-loop, we perform the same cross validation decoding procedure as above, but adding one extra procedure:
        #### shuffleing both the order of the feature matrix and the labels while the according mapping between the feature vectors and 
        #### the labels remains
        for ii in range(100):
            data,labels = utils.shuffle(data,labels)# the only difference from above
        idx = np.arange(data.shape[-1]).reshape(-1,50) # 28 by 50 matrix
#        cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=12345)# 5 fold cross validation
        clfs = []
        scores = []
        patterns = []
        for train,test in tqdm(cv.split(data,labels),desc='train-test'):# split the data into training set and testing set
            X = data[train]
            y = labels[train]
            # fit a classifier at each of the 50 ms window with only the training data and record the trained classifier
            clfs.append([make_clf(True).fit(X[:,:,ii],y) for ii in idx])
            # get the decoding pattern learned by each trained classifier at each of the 50 ms window with only the training data
            temp_patterns = np.array([get_coef(c,attr='patterns_',inverse_transform=True) for c in clfs[-1]])
            if temp_patterns.shape[-1] != 50:
                vecl = Vectorizer().fit(X[:,:,idx[0]])
                temp_patterns = np.array([vecl.inverse_transform(temp.reshape(1,-1))[0] for temp in temp_patterns])
            patterns.append(temp_patterns)
            X_ = data[test]
            y_ = labels[test]
            # compute the performance of each trained classifier at each of the 50 ms window with the testing data
            scores_ = [metrics.roc_auc_score(y_,clf.predict_proba(X_[:,:,ii])[:,-1]) for ii,clf in zip(idx,clfs[-1])]
            scores.append(scores_)
        scores = np.array(scores)
        patterns=np.array(patterns)
        plt.plot(scores.mean(0))
            
        pval = (np.array(chances.mean(1) > scores.mean(0)).sum(0)+1) / (n_perm +1) 
    
        results['scores_mean']=scores.mean(0)
        results['scores_std']=scores.std(0)
        results['chance_mean']=np.mean(chances,axis=1).mean(0)
        results['chance_se']=np.std(chances.mean(1))/np.sqrt(n_perm)# standard error
        results['clf']=clfs
        results['pval']=pval
        # average pattern learned by last dimension, which is the 50 ms window
        # average pattern learned by the classifier over 5 folds
        results['activity']=patterns.mean(-1).mean(0)
        pickle.dump(results,open(saving_dir+'temp_shuffle_%d (old vs new).p'%i_random,'wb'))
        results_.append(results)
    
    pickle.dump(results_,open(saving_dir+'shuffle results (old vs new).p','wb'))
    del results_    
    ##############################################################################################
    ##############################################################################################
    ####################### Temporal gnenralization   ############################################
    ###### train classifiers at each time sample and test the classifiers in other time samples###
    ###### this is going to take a long time to run ##############################################
#    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=12345)
#    clfs = [] # first we train 5 classifiers on each time point by a subset of the trials
#    for train,test in tqdm(cv.split(data,labels),desc='training'):
#        X = data[train]
#        y = labels[train]
#        clfs.append([make_clf().fit(X[:,:,ii],y) for ii in range(X.shape[-1])])
#        
#    scores_within = []# second, we test each 5 classifiers trained at a given time point at all possible time points
#    for fold,(train,test) in tqdm(enumerate(cv.split(data,labels)),desc='test within'):
#        X = data[test]
#        y = labels[test]   
#        scores_ = []
#        for clf in clfs[fold]:
#            scores_temp = [metrics.roc_auc_score(y,clf.predict_proba(X[:,:,ii])[:,-1]) for ii in range(X.shape[-1])]
#            scores_.append(scores_temp)
#        scores_within.append(scores_)
#    scores_within = np.array(scores_within)
#    
#    pickle.dump(scores_within,open(saving_dir+'temporal generalization(old vs new).p','wb'))
#    scores_within = pickle.load(open(saving_dir+'temporal generalization(old vs new).p','rb'))
#    
#    font = {
#            'weight' : 'bold',
#            'size'   : 20}
#    import matplotlib
#    matplotlib.rc('font', **font)
#    ### plot the temporal generalization 
#    fig,ax = plt.subplots(figsize=(12,10))
#    im = ax.imshow(scores_within.mean(0),# take the mean over the 5-fold cross validation
#                   origin='lower',aspect='auto',extent=[0,1400,0,1400],# some plotting thing
#                   cmap=plt.cm.RdBu_r,vmin=.5)# set the colormap and lowest value
#    cbar=plt.colorbar(im)
#    cbar.set_label('AUC')
#    ax.set(xlabel='Test time (ms)',ylabel='Train time (ms)',
#           title='Old vs New Temporal Generalization\nLinear SVM, 5-fold CV')
#    fig.savefig(saving_dir+'Old vs New decoding generalization.png',dpi=500)        



############### plot ######################################################################
########## you don't need to run anything above to plot if you have done that #############
from matplotlib import pyplot as plt
import pickle
import numpy as np
from glob import glob
working_dir = 'D:\\NING - spindle\\VCRT_study\\results\\'
shuffle_files = glob(working_dir+'*_shuffle*(old vs new).p')
results = [pickle.load(open(f,'rb')) for f in shuffle_files]
no_shuffle = results[0]
shuffle = results[1:] 
import mne
epochs = mne.read_epochs('D:\\NING - spindle\\VCRT_study\\data\\0.1-40 Hz\\new vs old-epo.fif',preload=False)
font = {
        'weight' : 'bold',
        'size'   : 20}
import matplotlib
matplotlib.rc('font', **font)
fig,ax = plt.subplots(figsize=(16,10))
times = np.linspace(25,1375,28)
ax.plot(times,no_shuffle['scores_mean'],color='k',alpha=1.,label='Classifi.Score (AUC Mean)_no shuffle')
m,s = np.array(no_shuffle['scores_mean']),np.array(no_shuffle['scores_std'])/np.sqrt(5)
ax.fill_between(times,m+s,m-s,color='red',alpha=.3,label='Classifi.Score (SE)')
ax.plot(times,no_shuffle['chance_mean'],color='k',linestyle='--',alpha=1.,label='Chance level (Mean)')
mm,ss = np.array(no_shuffle['chance_mean']),np.array(no_shuffle['chance_se'])
ax.fill_between(times,m+s,m-s,color='red',alpha=.7,lw=0.5)
for ii, item in enumerate(shuffle):
    if ii == 0:
        ax.plot(times,item['scores_mean'],color='blue',alpha=.7,label='Classifi.Score (AUC Mean)_shuffle')
    else:
        ax.plot(times,item['scores_mean'],color='blue',alpha=1.)
    m,s = np.array(item['scores_mean']),np.array(item['scores_std'])/np.sqrt(5)
    ax.fill_between(times,m+s,m-s,color='red',alpha=.3)
ax.set(xlabel='Time (ms)',ylabel='Classifi.Score (AUC)',title='Temporal Decoding\n Old vs New\nLinear SVM, 5-fold, n_permutation=1000',
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
fig.savefig('D:\\NING - spindle\\VCRT_study\\results\\'+'old vs new temporal decoding.png',dpi=500,bbox_inches = 'tight')  




sample_result = results[5]

times = np.vstack([np.arange(0,1450,50)[:-1],np.arange(0,1450,50)[1:]]).T
fig, axes = plt.subplots(figsize=(26,14),nrows=4,ncols=7)
k =5e-3
for idx,((start,stop),score,pvalue,ax,activity) in enumerate(zip(times,sample_result['scores_mean'],
                                                                sample_result['pval'],
                                                                axes.flatten(),
                                                                sample_result['activity'])):
    
    
    im,cn = mne.viz.plot_topomap(activity,epochs.info,axes=ax,show=False,vmin=-k,vmax=k)  
    if idx in pval_idx:
        im.axes.set(title = '%d-%d ms'%(start,stop),xlabel='%.2f*'%score)
    else:
        im.axes.set(title = '%d-%d ms'%(start,stop),xlabel='%.2f'%score)
fig.subplots_adjust(bottom=0.1, top=0.96, left=0.1, right=0.8,
            wspace=0.02, hspace=0.02)
# add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with 
# axes width 0.02 and height 0.8
cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
cbar = fig.colorbar(im, cax=cb_ax)
# set the colorbar ticks and tick labels
cbar.set_ticks([-k, 0, k])
cbar.set_ticklabels(['old image', 'no difference', 'new image'])
cbar.ax.set_title('         $\Delta$ $\mu$V = New - Old')
fig.savefig('D:\\NING - spindle\\VCRT_study\\results\\'+'old vs new topomap.png',dpi=500,bbox_inches='tight')

 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    