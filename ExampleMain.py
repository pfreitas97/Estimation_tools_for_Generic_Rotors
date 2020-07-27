#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:36:10 2020

@author: pedroaugustofreitasdearaujo
"""

import tensorflow as tf

import numpy as np

import pandas as pd

import os

# to obtain data
from Parser_UIUC_AeroData.propeller_data_util import merge_propeller_files

from rotor_ML_utils import get_UIUC_TrainingData, get_UIUC_RotorData

from rotor_ML_utils import hoverPerformance_Learned





import matplotlib.pyplot as plt







ROOT = os.path.join(os.getcwd(),"Parser_UIUC_AeroData")

VOL1PATH = os.path.join(ROOT,"Propeller_Data_V1")

VOL2PATH = os.path.join(ROOT,"Propeller_Data_V2")

SAVE_PATH_CT = os.path.join(os.getcwd(),"CT_MODEL")

SAVE_PATH_CQ = os.path.join(os.getcwd(),"CQ_MODEL")

#FILE_SAVE_PATH = os.path.join(SAVE_PATH,"testModel.h5")



DF1 = merge_propeller_files(VOL1PATH)

DF2 = merge_propeller_files(VOL2PATH)





naca4412 = pd.read_csv(os.path.join("Testing_Airfoils","NACA4412_RE500000.txt"),delim_whitespace=True)

# Scrapping unnecessary data

naca4412 = naca4412.iloc[1:,:3]








x_dataset,y_dataset = get_UIUC_TrainingData([DF1,DF2],rescale_coefficients=True,airfoil=naca4412) 




model_CT = tf.keras.Sequential([
    
    tf.keras.layers.Dense(units=49,input_shape=[49,]),
    tf.keras.layers.Dense(units=100,activation='relu'),
    tf.keras.layers.Dense(units=100,activation='relu'),
    tf.keras.layers.Dense(units=1)  
    ])

model_CQ = tf.keras.Sequential([
    
    tf.keras.layers.Dense(units=49,input_shape=[49,]),
    tf.keras.layers.Dense(units=100,activation='relu'),
    tf.keras.layers.Dense(units=100,activation='relu'),
    tf.keras.layers.Dense(units=1)  
    ])


optim = tf.keras.optimizers.RMSprop()

# Compiling

model_CT.compile(loss='mse',optimizer=optim,metrics=['mae','mean_absolute_percentage_error'])

model_CQ.compile(loss='mse',optimizer=optim,metrics=['mae','mean_absolute_percentage_error'])

# # Note last two columns are the dependent values
full_Data = np.concatenate([x_dataset,y_dataset],axis=1)

np.random.shuffle(full_Data)

trainRatio = 0.85
nRows = full_Data.shape[0]

train,test = full_Data[:round(trainRatio*nRows)], full_Data[round(trainRatio*nRows):]

EPOCHS = 300


x_train = train[:,:49]

y_train = train[:,49:]

x_test = test[:,:49]

y_test = test[:,49:]


history_CT = model_CT.fit(x_train,y_train[:,0],epochs=EPOCHS)

history_CQ = model_CQ.fit(x_train,y_train[:,1],epochs=EPOCHS)


model_CT.save(SAVE_PATH_CT)

model_CQ.save(SAVE_PATH_CQ)




# testResults = model.evaluate(x=test[:,:49],y=test[:,49:])



# plt.plot(np.arange(EPOCHS),history.history['mae'])



# model.save(SAVE_PATH)

rotorTestData = get_UIUC_RotorData([DF1,DF2])

row = 0


b = rotorTestData[row][0]
R = rotorTestData[row][1]
C = rotorTestData[row][2]
omega = rotorTestData[row][3]
rR =  np.array(rotorTestData[row][4])
cR = np.array(rotorTestData[row][5])
twist = np.array(rotorTestData[row][6])
CT_Test = rotorTestData[row][7]
CQ_Test = rotorTestData[row][8]






CT_learned,CQ_learned = hoverPerformance_Learned(b,R,C,omega,naca4412,rR,cR,twist,SAVE_PATH_CT,SAVE_PATH_CQ)



# print(testResults[1])

