# VCRT_study

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
![new vs old](https://github.com/adowaconan/VCRT_study/blob/master/results/old%20vs%20new%20temporal%20decoding.png)


## Temporal generalization: trained at a given segment and tested at all segments
