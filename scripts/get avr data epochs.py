# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 12:57:22 2018

@author: ning
"""

import os
os.chdir('D:\\NING - spindle\\VCRT_study\\scripts')
import avr_reader 
import mne
import numpy as np
from glob import glob

news = glob('D:\\NING - spindle\\VCRT_study\\data\\new\\*.avr')
olds = glob('D:\\NING - spindle\\VCRT_study\\data\\old\\*.avr')
scrs = glob('D:\\NING - spindle\\VCRT_study\\data\\scr\\*.avr')

data = []
for f in news:
    temp = avr_reader.avr(f)
    channelNames = temp['channelNames'][:-3]
    data.append(temp['data'][:-3])
info = mne.create_info(ch_names=channelNames,sfreq = 1000,ch_types='eeg')#montage='standard_1020')
data = np.array(data)
data = data / 1e6
new = mne.EpochsArray(data,info)
new.set_montage(mne.channels.read_montage('standard_1020'))
new.event_id = {'new':1}
new.events[:,-1]=1   

data = []
for f in olds:
    temp = avr_reader.avr(f)
    channelNames = temp['channelNames'][:-3]
    data.append(temp['data'][:-3])
info = mne.create_info(ch_names=channelNames,sfreq = 1000,ch_types='eeg')#montage='standard_1020')
data = np.array(data)
data = data / 1e6
old = mne.EpochsArray(data,info)
old.set_montage(mne.channels.read_montage('standard_1020'))
old.event_id = {'old':0}
old.events[:,-1]=0  

data = []
for f in scrs:
    temp = avr_reader.avr(f)
    channelNames = temp['channelNames'][:-3]
    data.append(temp['data'][:-3])
info = mne.create_info(ch_names=channelNames,sfreq = 1000,ch_types='eeg')#montage='standard_1020')
data = np.array(data)
data = data / 1e6
scr = mne.EpochsArray(data,info)
scr.set_montage(mne.channels.read_montage('standard_1020'))
scr.event_id = {'scrs':2}
scr.events[:,-1]=2

epochs = mne.concatenate_epochs([new,old])
epochs.save('D:/NING - spindle/VCRT_study/data/0.1-40 Hz/new vs old-epo.fif')

epochs = mne.concatenate_epochs([new,old,scr])
epochs.save('D:/NING - spindle/VCRT_study/data/0.1-40 Hz/3 classes-epo.fif')
