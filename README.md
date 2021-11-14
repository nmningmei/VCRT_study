# VCRT_study

## Requirement
- Python 3.7.10
- mne=0.23.4
- numpy=1.21.2
- scipy=1.5.3
- matplotlib=3.4.3
- scikit-learn=1.0
- numba=0.53.1
- pandas=1.3.3

## Two experimants:
### Old image vs New image
### Old image vs New Image vs Scramble image

## classifier: MNE-python vectorizer - Standardizer - Linear SVM
## 5-fold stratified cross validation

## data dimension: 
### 54 by 61 by 1400 (ERP, EEG channel, time)
### 80 by 61 by 1400 (ERP, EEG channel, time)

## after train-test split, data is split by a 50 ms window along the last dimension
### training set: 42 by 61 by 50 by 28 -------- testing set: 12 by 61 by 50 by 28
### training set: 64 by 61 by 50 by 28 -------- testing set: 16 by 61 by 50 by 28
## classifiers were trained ant tested along the last dimension (5 * 28 classifiers were trained and tested)

## Temporal decoding: trained and tested at the same segment 
### old vs new
![old vs new](https://github.com/adowaconan/VCRT_study/blob/master/results/old%20vs%20new%20temporal%20decoding.png)
### old vs new topograph maps:
![old vs new topomap](https://github.com/adowaconan/VCRT_study/blob/master/results/old%20vs%20new%20topomap.png)
### old vs new vs scramble
![old vs new vs scramble](https://github.com/adowaconan/VCRT_study/blob/master/results/old%20vs%20new%20vs%20scr%20temporal%20decoding.png)

## Temporal generalization: trained at a given segment and tested at all segments
### old vs new
![old vs new](https://github.com/adowaconan/VCRT_study/blob/master/results/Old%20vs%20New%20decoding%20generalization.png)
### old vs new vs scramble
![old vs new vs scramble](https://github.com/adowaconan/VCRT_study/blob/master/results/Old%20vs%20New%20vs%20scr%20decoding%20generalization.png)
