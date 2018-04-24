# -*- coding: utf-8 -*-
"""
Created on Fri May  5 13:41:54 2017

@author: ning mei

The function (avr) takes .avr files and return attributions in the file. 
Translation from BESA Matlab functions.
"""

import numpy as np
def avr(filename):
    result = {'data':[]}
    with open(filename) as f:
        for ii, line in enumerate(f.readlines()):
            if ii == 0:
                attrs = [a[:-1] for a in line.split(' ') if ('=' in a)]
                attrs = attrs[:-1]
                values=[]
                for text in line.split(' '):
                    try:
                        temp_value = float(text)
                        values.append(temp_value)
                    except:
                        pass
                for jj,attr in enumerate(attrs):
                    result[attr] = values[jj]
            elif ii == 1:
                result['channelNames'] = line[1:-2].split(' ')
            else:
                result['data'].append([float(v) for v in line.split(' ') if (len(v) > 0)])
    result['data'] = np.array(result['data'])
    return result
                