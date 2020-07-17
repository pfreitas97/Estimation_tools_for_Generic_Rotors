#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:36:10 2020

@author: pedroaugustofreitasdearaujo
"""

import tensorflow as tf

import numpy as np

import pandas as pd






model = tf.keras.Sequential([
    
    tf.keras.layers.Dense(units=34,input_shape=[34,]),
    tf.keras.layers.Dense(units=34),
    tf.keras.layers.Dense(units=34),
    tf.keras.layers.Dense(units=34),
    tf.keras.layers.Dense(units=4)  
    ])


