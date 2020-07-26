#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 10:33:56 2020

@author: pedroaugustofreitasdearaujo
"""

import tensorflow as tf

import numpy as np

import pandas as pd

import rotorHoverPerformance as rp

from Parser_UIUC_AeroData.UIUC_Propeller import UIUC_Propeller

from Parser_UIUC_AeroData.UIUC_Propeller import _rescaleLinearly





'''TODO: TEST '''

def _convert_to_nn_vector(b,R,C,omega,rR,cR,twist,scale_omega,blade_elem_length):
    ''' Handles the conversion from the format typically used by the hover performance functions to the one expected by
    the neural network trained. Among other things it handles the conversion to a single (1,*) numpy array, 
    as well as the rescalling of the omega parameter
    
    Parameters:
        b: blade number
        R: Radius
        C: Chord
        omega: Angular velocity
        r/R: Blade element position as a function of max Radius
        c/R: Chord as a function of blade element positon 
        twist: Angle of twist at each blade element point [expected in Radians]
        scale_omega: the scalar by which every omega value was divided during training
    '''
    
    ''' NOTE: This was an assumption I made about how this function would be used. Should be a fairly easy fix if you'd
    like to pass a vector of angular velocities instead '''
    
    
    rescaled_rR = _rescaleLinearly(rR, blade_elem_length)
    
    rescaled_cR = _rescaleLinearly(cR, blade_elem_length)
    
    rescaled_twist = _rescaleLinearly(twist, blade_elem_length)
    
    rescaled_omega = omega/scale_omega
    
    num_rows = 1
    
    num_columns = 4 + 3*blade_elem_length
    
    nn_vector = np.zeros((1,num_columns))
    
    if np.isscalar(rescaled_omega):
        nn_vector = np.array([b,R,C,rescaled_omega,*rescaled_rR,*rescaled_cR,*rescaled_twist])
        
        nn_vector = nn_vector.reshape(1,49)
        

    else:
        num_rows = max(omega.shape)
        nn_vector =  np.zeros((num_rows,num_columns))
        
        for shape, curr_omega in np.ndenumerate(rescaled_omega):
            
            nn_vector[max(shape),] = np.array([b,R,C,curr_omega,*rescaled_rR,*rescaled_cR,*rescaled_twist])
            
            
            pass
        
        '''TODO decide if numpy array needs reshaping here '''
        
        #nn_vector = nn_vector.reshape(num_rows,num_columns)
        pass
    
    
    return nn_vector





def get_UIUC_TrainingData(dataframes,rescale_coefficients=False,airfoil = None,blade_elements=15,scale_omega=3000,
                         dropDuplicates=True,hoverFunction = rp.hoverPerformance_NoHeuristics):
    ''' Parse through iterable (list,tuple) of dataframes where every row follows the format 
    [propller_name,diameter,pitch,geometric_path,static_path] as seen in 'merge_propeller_files' within the 
    Parse_UIUC_Aerodata module. 
    Return two numpy matrices. First matrix contains the indpendent variables and every row follows the format below:
    [blade number, radius, max chord, rot. speed[rad/s], *r/R , * c/R, *twist] (where *variable implies values are tabulated 
    at specific blade element points). 
    
    The second matrix contains the dependent variables: [CT,CQ], the coefficients of thrust and torque respectively. 
    
    Parameters:
    
    dataframes: iterable of dataframes following format described above
    
    rescale_coefficients: When set to true, calculate the coefficients through momentum theory and divide the result
    
    airfoiil: dataframe with airfoil information [alphas,CL,CD] (currently assuming all propellers utilize the same airfoil)

    
    blade_element: determines number of tabulated blade elements and effectvely the dimensions of the 
    independent variable arrays.
    
    scale_omega: In order to maintain all parameters around the same order the angular velocity should be divided by 
                    something, one possible solution is the maximum value in the test data or a value around that value. 
                    It is important to remember to scale omega by the same value if using the trained network to predict.
                    
    dropDuplicates: reduce number of different propeller files used that may be utilizing some of the same files.
    
    hoverFunction: Function that calculates the thrust and power coefficients using geometric / airfoil information.
                    By default, it utilizes a barebones version of the function without any approximating heuristics 
                    based on flow/ hub conditions.
    '''
    
    assert(len(dataframes) > 0)
    
    df = pd.concat(dataframes, ignore_index=True)
    
    XMAT = [] # Full matrix with every individual training data file vertically concatenated
    YMAT = []
    
    for index,row in df.iterrows():
        
        curr_prop = UIUC_Propeller(row)
        
        xmat,ymat = curr_prop.getTrainingData()
        
        
        # The fourth column contains the angular velocity information (omega)
        
        assert(np.isscalar(scale_omega) and scale_omega > 0)
        
        xmat[:,3] = np.divide(xmat[:,3],scale_omega)
        
        if rescale_coefficients:
            
            assert(airfoil is not None)
            
            propList = curr_prop.getRotorData()
            
            b = propList[0]
            R = propList[1]
            C = propList[2]
            omega = propList[3]
            rR = propList[4]
            cR = propList[5]
            twist = propList[6]
            
            
            CT,CQ = hoverFunction(b,R,C,omega,airfoil,rR,cR,twist)
            
            ymat[:,0]= np.divide(ymat[:,0],CT)
            
            ymat[:,1] = np.divide(ymat[:,1],CQ)
            
            
            pass
        
        
        
        if index == 0:
            XMAT = xmat
            YMAT = ymat
            pass
        else:
            XMAT = np.concatenate([XMAT,xmat])
            YMAT = np.concatenate([YMAT,ymat])            
            pass
        
        
        
        pass
    
    return [XMAT,YMAT]

def get_UIUC_RotorData(dataframes):
    '''Parse through iterable (list,tuple) of dataframes where every row follows the format 
    [propller_name,diameter,pitch,geometric_path,static_path] as seen in 'merge_propeller_files' within the 
    Parse_UIUC_Aerodata module. 
    
    Return List of Lists. 
    
    The first is a list of lists with the following format:
    [Rotor_data_0,Rotor_data_1,...] wheree Rotor_Data is also a list with the format:
    Rotor_data = [b,R,C,Omega [rad/s], rR, cR, twist, CT,CQ]. Note that rR,cR,twist are iterables of length > 1. 
    Additionally CT,CQ are the independent variables
    
    '''
    assert(len(dataframes) > 0)
    
    df = pd.concat(dataframes, ignore_index=True)
    
    dataList = [] # Full matrix with every individual test file vertically concatenated
    
    
    
    ''' TODO CHANGE BEHAVIOUR TO OUPUT ONE LIST ELEMENT FOR EVERY ANGULAR VELOCITY FOR BOTH X AND Y '''
    
    for index, row in df.iterrows():
        
        curr_prop = UIUC_Propeller(row)
        
        propList = curr_prop.getRotorData() # this list includes all omega data
        
        _ ,y = curr_prop.getTrainingData() # this is a ndarray with every ct,cq for each rpm
        
        # here i must go through each omega in prop list, and append the row using omega as a simple scalar
        
        print(y)
        
        b = propList[0]
        R = propList[1]
        C = propList[2]
        omega = propList[3]
        rR = propList[4]
        cR = propList[5]
        twist = propList[6]
        
        #print(np.isscalar(omega.shape))
        
        for shape,curr_omega in np.ndenumerate(omega):
            
            # Since y is returned as an [n,2]  array each row must be separated individually
            
            CT = y[max(shape),0] #[CT(curr_omega),CQ(curr_omega)]
            CQ = y[max(shape),1]
            
            row = [b,R,C,curr_omega,rR,cR,twist,CT,CQ]
                        
            dataList.append(row)
        
            pass
        
        pass
    
    
    return dataList




def hoverPerformance_Learned(b,R,C,omega,airfoil,rR,cR, twist,saved_tf_path_CT,saved_tf_path_CQ,
                             scale_omega=3000,blade_elem_training=15):
    ''' Main function used to compute CT and CQ based a learned heuristic for CT and CQ adjustment
    
    Params:
        b: Number of Blades (integer)
        
        R: Radius in [Meters]
        
        C: Chord [Meters] if varying chord please add maximum here and a discrete number of points in the cR variable
        
        omega: Rotation rate in [rad/s]
        
        r/R: (0,1] np.array with Position of blade elements as a function of Radius, needed when the chord is not constant
              
        c/R: Chord length divided by Radius, please make sure r/R and c/R refer to the same points
        
        twist: Either a vector with the twist at every blade element or the minimum and maximum if using linear twist. 
        Twist should be passed in Degrees.
        
        airfoil: Functionality is still being added. Currently it is running with
        a Pandas dataframe [alpha,cl,cd] of your desired airfoil
        
        saved_tf_path_CT: Path for the trained model predicting CT weight
        
        saved_tf_path_CQ: Path for the trained model predicting CQ weight
        
        scale_omega: the scalar by which every omega value was divided during training
        
        blade_elem_training: number of blade elements (and by consequence the dimensions) in the trained model
    '''
    
    #RHO = 1.225 #[kg/m3]
    
    assert(len(rR) == len(cR))
    
    assert(len(rR) == len(twist) or len(twist) == 2 or len(twist) == 0)
    
    assert(scale_omega > 0)
    
    twist_provided = True
    
    
    # Tabulating elements if they were not provided
    if len(rR) == 0:
        rR,cR,twist = rp._tabulateBladeElements(C, R, twist)
        twist_provided = False
        pass
    
    # Making sure all iterables are in a format that is suitable:
    
    rR = np.array(rR)
    cR = np.array(cR)
    twist = np.array(twist)
    
    
    
    # Converting twist from Degrees to Radians
    
    twist = twist * (np.pi)/180
    
    
    
    
    alphaZL = rp._getAlphaZL(airfoil)
    
    a = rp._getLiftSlopeCurve(airfoil, omega, R, rR)

    
    # step 4 Obtaining theta from twist and zero angle of attack 
    
    theta = rp._getCollectivePitch(twist, alphaZL,twist_provided)
    
    # Step 5
    inducedAngle = rp._getInducedVelocity(a, b, cR, rR, theta)
    
    
    
    # Step 6
    
    alphas = rp._getLocalAOA(theta, inducedAngle)
    
    
    # Step 7 (Get cl and cd for every angle of attack on preceding list)
        
    aeroCoef_df = rp._extractClandCd(airfoil, alphas)
    
    cl = aeroCoef_df['CL'].to_numpy(dtype=float)
    
    cd = aeroCoef_df['CD'].to_numpy(dtype=float)
        
    # Step 8    
    
    runningThrustLoad = rp._getRunningThrustLoading(b,rR,cR,cl)
    
    
    # Step 9
    
    ''' Steps 10,11 have been excluded in this barebone version. CT returned does not include any 
    attempts to correct for tip loss / hub loss. Hub position is assumed to be zero, and the CT returned is modified only 
    by the prediction of the first ouput node in the tensorflow model provided.'''
    
    CTnoLoss = rp._getCTnoTipLoss(runningThrustLoad, rR, x_hub=0)
    
    
    # Step 10 (skipped calculation of tip loss region)    
        
    # Step 11 (skipped calculation of expected tip loss)
    
    ''' Tip loss is explicitly ignored here, relying on the learned weight'''
    
    CT = CTnoLoss
    
    
    # Step 12
    
    runProfile = rp._getRunningProfileTorqueCoefficient(b, rR, cR, cd)
    
    
    # Step 13
    
    CQ0 = rp._getProfileTorqueCoefficient(runProfile,rR)
    
    
    # Step 14
    
    runningInducedTorque = rp._getRunningInducedTorqueCoefficient(b,rR,cR,cl,inducedAngle)
    
    
    #Step 15
    
    ''' B is set to 1, and x_hub is set to 0 in order to remove tip loss and rotor hub effects in return value'''
    
    CQi = rp._getInducedTorqueCoefficient(runningInducedTorque,rR,B=1,x_hub=0)
        
    
    #Steps 16 17 and 18 are only needed when using the default heuristic

    # Step 16 (skipped computing delQ, swirl induced torque)
            
    # Step 17 (skipped computing Disk Loading)
        
    # Step 18 (skipped determining Disk Loading / solidity ratio)
    
    # Step 19 (skipped computing wake factor)
    
    
    # Step 20
    
    CQ = CQ0 + CQi
    
    
    
    
    if not np.isscalar(CQ):
        CT = np.full(CQ.shape ,CT)
        pass
    
    loaded_CT_model = tf.keras.models.load_model(saved_tf_path_CT)
    
    loaded_CQ_model = tf.keras.models.load_model(saved_tf_path_CQ)
    
    
    
    target_vector = _convert_to_nn_vector(b,R,C,omega,rR,cR,twist,scale_omega,blade_elem_training)
    
    
    nn_weight_CT = loaded_CT_model.predict(target_vector)
    
    nn_weight_CQ = loaded_CQ_model.predict(target_vector)
    
    print("nn_weight_ct %s" % nn_weight_CT)
    
    print("nn_weight_cq %s" % nn_weight_CQ)
    
    
        
    return np.array([nn_weight_CT * CT,nn_weight_CQ*CQ])

