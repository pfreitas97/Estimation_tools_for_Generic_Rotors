#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 11:16:06 2020

@author: pedroaugustofreitasdearaujo
"""


import numpy as np

import scipy.integrate


# For more accurate estimation of the effect of a hub and tip loss
import scipy.interpolate


# Testing regression models for airfoil slope

import scipy.stats




#needed for pulling files from UIUC database

import os

import pandas as pd




# For testing right now

import matplotlib.pyplot as plt



''' Note REALLY NEED TO COMPLETE THE TEST CASES FOR BASIC FUNCTIONS'''


# Class is meant to store geometry of a specific rotor and perform certain tasks

# Maybe functions first would be more illuminating

# class RotorObj:
    
#     def __init__(self):
        
#         #b,R,C,omega,theta0,cl_cd_alpha) 
        
#         self.name = []
        
#         self.




# This name will be change when refactoring most likely

# DO'S: final place where all calculations are made, can easily be modified to return
# Desired information that was not explicitly included.

# Takes in: (b) number of blades, R = Radius (m), C = chord (m), 
# Omega = Angular V (rad/s), theta0 

#def rotorPerformanceFromGeometry(b,R,C,omega,theta0, thetaMin, cl_cd_alpha):
    

    
    
# Step 3 Tabulate Lift Slope Curve:

def _tabulateBladeElements(C,R,twist,n = 10):
    ''' Creates a list of blade elements for integration if the number was not specified by the user. 
    Assumes chord is constant since otherwise the blade elements would have to be specified by the user.
    '''
    
    rR = np.linspace(1/n ,1 ,n)
    
    cR = np.full((n,1),C/R)
    
    assert(len(twist) == 2 or len(twist) == 0)
    
    
    
    if len(twist) == 2:
        twist = np.linspace(twist[0],twist[1],n) # Note it assumes linear twist
    else:
        twist = np.full((n,1),0)
    
    return [rR, cR, twist]



# Estimate lift slope curve:

    
''' TODO Change lift slope curve to utilize airfoil information and Mach Number info'''
def _getLiftSlopeCurve(airfoil,omega,R,rR):
    ''' Extrapolates the effective lift slope curve for every blade element based on Mach number and airfoil shape'''
    
    # Thin airfoil approximation:
    #np.full(rR.shape,2*np.pi*(np.pi/180))
    
    y = airfoil['CL'].to_numpy(dtype=float)
    x = airfoil['alpha'].to_numpy(dtype=float)
    
    
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)
    
    #print('slope: %s' % slope)
    #print('intercept: %s' % intercept)
    #print('rvalue: %s' % r_value)
    
    return np.full(rR.shape,slope)






# Step 4 choose collective pitch (if it is not given)


''' TODDDOOOOOO CREATE better description and function'''

def _getCollectivePitch(twist,alphaZL):
    ''' Take local twist and airfoil zero lift angle in Rad and return a list that guarantees the 
    effective angle is never negative. Here collective pitch is chosen to be equal to the zero lift angle. 
    This can
    '''
    
    collective = alphaZL
    
    
    return collective + twist - alphaZL


''' '''



# Step 5


# get inflow angle and induced velocity

# Function takes in:
# a:  lift slope curve
# b:   number of blades b,
# r/R:  a vector of elements displaying the ratio of the blade element to the max radius
# c/R : a vector of elelements with the ratio of the local chord to the maximum radius
# theta: a vector of the local angle 

#returns a vector of the induced velocity of every point



def _getInducedVelocity(a,b, cR, rR,theta):
    '''Return inflow angle and induced velocity ratio.
    
    Parameters:
    a:  lift slope curve (Cl per RADIANS)
    b:   number of blades b
    r/R:  a vector of elements displaying the ratio of the blade element to the max radius (0,1]
    c/R : a vector of elelements with the ratio of the local chord to the maximum radius
    theta: a vector of the local angle (RADIANS)
    '''
        
    assert(np.isscalar(theta) or len(rR) == len(theta))
    
    assert(np.isscalar(cR) or len(cR) == len(rR))
    
    
    return  ((a * b * cR) / (16 *np.pi* rR)) * (-1 + np.sqrt(1 + (32 * np.pi * theta * rR) /(a * b * cR)  ) )
    








# Step 6


# theta and induced angle will be in RADIANS

# but the angle of attack returned will be in DEGREES due to historical convention for AOA (and the way xfoil tabulates data)

def _getLocalAOA(theta, inducedAngle):
    
    return (theta - np.arctan(inducedAngle)) * (180/np.pi) 



''' TODO  FINISH / TEST '''
# Step 7 -- extract cl,cd as a functon of mach number and airfoil Shape 


def _extractClandCd(airfoil,alphas):
    
    cl_list = []
    cd_list = []
    
    cl_column = airfoil.columns.get_loc('CL')
    cd_column = airfoil.columns.get_loc('CD')
    
    for alpha in alphas:
        
        index = abs(airfoil['alpha'].to_numpy(dtype=float) - alpha).argmin()
        
        cl = airfoil.iloc[index,cl_column]
        
        cd = airfoil.iloc[index,cd_column]
                
        cl_list.append(cl)
        
        cd_list.append(cd)
        
        
        pass
    
    df = pd.DataFrame({'CL' : cl_list, 'CD' : cd_list})
    
    return df

    




#NACA0012 = pd.read_csv("NACA0012_RE500000_RAWXF.txt",delim_whitespace=True)


# Step 8 

# cl is assumed to be a numpy array of the appropriate length obtained in step 7

def _getRunningThrustLoading(b,rR,cR,cl):
    
    assert(len(cl) == len(rR))
    
    
    return ((b * (rR**2) * cR * cl)/(2*np.pi))




# Step 9

# Integrate removing the assumed hub radius 


def _getCTnoTipLoss(dct_dr, rR, x_hub = 0.1):
    
    # find index of every element larger than the hub
    
    assert(len(dct_dr) == len(rR))
    
    tck = scipy.interpolate.splrep(rR,dct_dr)
    
    #ftest = scipy.interpolate.interp1d(rR,dct_dr,fill_value="extrapolate")
    
    
    ## Test by plotting
    
    #xt = np.linspace(0,1,100)
    
    #yt = scipy.interpolate.splev(xt,tck)
    
    #yt = ftest(xt)
    
    #plt.plot(rR,dct_dr,'o',xt,yt)
    
    #integral,err = scipy.integrate.quad(ftest,x_hub,1,limit=80)
    
    
    
        
    ## Prev method that is quieter (TO DELETE?)
    
    # print("quad result: %s" % integral)
    
    # print("splint result: %s "% splintegral)
    

    # indexs = [i for i,l in enumerate(rR) if l > x_hub]
    
    # rR_toIntegrate = rR[indexs]
    
    # dct_dr_toIntegrate = dct_dr[indexs] 
    
    # scipy.integrate.simps(dct_dr_toIntegrate,rR_toIntegrate)
    
    # scipy.interpolate.splint(x_hub,1,tck)
    
    return scipy.interpolate.splint(x_hub,1,tck)





# Step 10

'''TODOO make necessary changes to address the other two alternatived heuristics'''

def _getTipLossFactor(CTnoTipLoss,b):
    
    assert(np.isscalar(CTnoTipLoss))

    assert(np.isscalar(b))
    
    assert(b > 0)
    
    return 1 - ((np.sqrt(2*CTnoTipLoss))/b)






# Step 11

def _getCorrectedCT(CTnoTipLoss, dct_dr, rR, B = 0.91):
    
    assert(len(dct_dr) == len(rR))
    
    
    tck = scipy.interpolate.splrep(rR,dct_dr)
        
    
    # ftest = scipy.interpolate.interp1d(rR,dct_dr,fill_value="extrapolate")
    
    
    # integral,err = scipy.integrate.quad(ftest,B,1,limit=80)
    
    
    
    # print("Corrected CT: quad result: %s" % integral)
    
    # print("Corrected CT: splint result: %s "% splintegral)
    
    
    return CTnoTipLoss - scipy.interpolate.splint(B,1,tck)

# step 12
    
def _getRunningProfileTorqueCoefficient(b,rR, cR, cd):
    
    assert(len(rR) == len(cd))
    assert(len(rR) == len(cR))
    
    
    return (b * (rR**3) * cR * cd) / (2*np.pi)


# Step 13

''' TODOOOO Check which integration is more accurate when continuous func not used '''

    
def _getProfileTorqueCoefficient(dq_dr,rR):
    
    assert(len(dq_dr) == len(rR))
    
    tck = scipy.interpolate.splrep(rR,dq_dr)
        
    
    # ftest = scipy.interpolate.interp1d(rR,dq_dr,fill_value="extrapolate")
    
    
    # integral,err = scipy.integrate.quad(ftest,0,1,limit=80)
    
    # print("Profile Splint: %s" % splintegral)
    
    # print("Profile quad: %s" % integral)
    
    return scipy.interpolate.splint(0,1,tck)


# Step 14

def _getRunningInducedTorqueCoefficient(b,rR,cR,cl,induced_Vel):
    
    
    
    return (b *  (rR**3) * cR * cl * induced_Vel)/ (2*np.pi)




# Step 15
def _getInducedTorqueCoefficient(dqi_dr,rR, B = 0.9, x_hub = 0.1):
    
    assert(len(dqi_dr) == len(rR))
    

    tck = scipy.interpolate.splrep(rR,dqi_dr)
    
    # splintegral = scipy.interpolate.splint(x_hub,B,tck)
    
    
    # Code integrating with Gaussian quadrature, it is more accurate but usually does not really make a difference
    
    # ftest = scipy.interpolate.interp1d(rR,dqi_dr,fill_value="extrapolate")
    
    # integral,err = scipy.integrate.quad(ftest,0,1,limit=80)
    
    
    return scipy.interpolate.splint(x_hub,B,tck)

# Step 16

'''TODO Address inherent problems in this heuristic 
        and make something thats better'''
        
        
''' HEURISTIC TO ESTIMATE '''


def _computeDelCQ(CT,CQi):
    
    assert(np.isscalar(CT))
    assert(np.isscalar(CQi))

    
    ct_candidates =  np.linspace(0,0.05,11)
    
    
    
    #From linear correction figure 1.29
    
    linearCorrection = np.array([0, 0.011, 0.028, 0.04, 0.053, 0.064, 0.075, 
                                0.083, 0.092, 0.10, 0.11])
    
    idx = np.abs(ct_candidates - CT).argmin()
    
    return linearCorrection[idx]*CQi




# Step 17

def _computeDiskLoading(CT,rho,omega,R):
    
    return (CT * rho *( omega*R)**2)






# Step 18

def _computeRatioCTtoSolidity(CT,b,C,R):
    
    return (CT)/(((b*C)/(np.pi*R)))


# Step 19

''' HEURISTIC TO ESTIMATE '''

def _computeWakeTorqueLinearCorrection(diskLoading,CT_solidity):
    # Source Figure 1.34
    
    '''TODO CREATE A CORRECTION THAT IS BETTER'''
    
    return 0.94 + (1.06 * diskLoading*CT_solidity)




# Step 20

def _computeTorqueCoefficient(CQ,CQi,delCQ,wakeCorrection):
    
    assert(np.isscalar(CQ))
    
    assert(np.isscalar(CQi))

    assert(np.isscalar(delCQ))

    assert(np.isscalar(wakeCorrection))
    
    return (CQ + CQi + delCQ)*wakeCorrection







def HoverPerformance_Classical(b,R,C,omega,airfoil,rR = [],cR = [], twist = []):
    ''' Main function to compute the Coefficient of Thrust and Torque for a hovering aircraft
    
    Params:
        b: Number of Blades (integer)
        
        R: Radius in [Meters]
        
        C: Chord [Meters] if varying chord please add maximum here and a discrete number of points in the cR variable
        
        omega: Rotation rate in [rad/s]
        
        r/R: (0,1] np.array with Position of blade elements as a function of Radius, needed when the chord is not constant
              
        c/R: Chord length divided by Radius, please make sure r/R and c/R refer to the same points
        
        twist: Either a vector with the twist at every blade element or the minimum and maximum if using linear twist. 
        Please Use DEGREES here, internally computations will be done in 
        
        airfoil: Functionality is still being added. Currently it is running with 
        Pandas dataframe [alpha,cl,cd] of your desired airfoil. Alpha is expected in degrees
    '''
    
    RHO = 1.225
    
    assert(len(rR) == len(cR))
    
    
    assert(len(rR) == len(twist) or len(twist) == 2 or len(twist) == 0)
    
    
    # Tabulating elements if they were not provided
    if len(rR) == 0:
        rR,cR,twist = _tabulateBladeElements(C, R, twist)
        pass
    
    
    # Converting twist from Degrees to Radians
    
    twist = twist * (np.pi)/180
    
    
    
    # Obtaining theta from twist and zero angle of attack 
    '''ASSUMING ZERO ANGLE OF ATTACK COMPUTED ALREADY PLEASE ADD CODE HERE ONCE IT IS DONE '''
    
    alphaZL = 0
    
    theta = _getCollectivePitch(twist, alphaZL)
        
    
    # Get lift slope curve
    ''' TODO MAKE ACCURATE '''
    
    
    # Note make sure this returns either a pandas series or a numpy ndarray
    a = _getLiftSlopeCurve(airfoil, omega, R, rR)
    
    
    
    # Step 5
    inducedAngle = _getInducedVelocity(a, b, cR, rR, theta)
    
    
    
    # Step 6
    
    alphas = _getLocalAOA(theta, inducedAngle)
    
    
    # Step 7 (Get cl and cd for every angle of attack on preceding list)
    
    
    '''refactor for efficiency and clarity '''
    
    aeroCoef_df = _extractClandCd(airfoil, alphas)
    
    cl = aeroCoef_df['CL'].to_numpy(dtype=float)
    
    cd = aeroCoef_df['CD'].to_numpy(dtype=float)
        
    # Step 8    
    
    runningThrustLoad = _getRunningThrustLoading(b,rR,cR,cl)
    
    
    # Step 9
    
    CTnoLoss = _getCTnoTipLoss(runningThrustLoad, rR)
    
    print("CT no loss: %s" % CTnoLoss)

    
    # Step 10
    
    B = _getTipLossFactor(CTnoLoss,b)
    
    
    print("B: %s " % B)
    
    # Step 11
    
    CT = _getCorrectedCT(CTnoLoss, runningThrustLoad, rR, B)
    
    print("CT: %s" % CT)

    
    
    # Step 12
    
    runProfile = _getRunningProfileTorqueCoefficient(b, rR, cR, cd)
    
    
    
    # Step 13
    
    CQ0 = _getProfileTorqueCoefficient(runProfile,rR)
    
    print("CQ0: %s" % CQ0)
    
    # Step 14
    
    runningInducedTorque = _getRunningInducedTorqueCoefficient(b,rR,cR,cl,inducedAngle)
    
    
    #Step 15
    
    CQi = _getInducedTorqueCoefficient(runningInducedTorque,rR)
    
    print("CQi: %s" % CQi)
    
    # Step 16
    
    
    delCQ = _computeDelCQ(CT,CQi)
    
    
    
    # Step 17
    
    diskLoading = _computeDiskLoading(CT,RHO,omega,R)
    
    
    # Step 18 
    
    ''' CALCULATED FOR A HEURISTIC'''
    
    CT_div_sigma = _computeRatioCTtoSolidity(CT,b,C,R)
    
    # Step 19
    
    ''' HEURISTIC '''
    
    wakeFactor = _computeWakeTorqueLinearCorrection(diskLoading,CT_div_sigma)
    
    
    # Step 20
    
    CQ = _computeTorqueCoefficient(CQ0,CQi,delCQ,wakeFactor)
    
    print("CQ: %s" % CQ)
    
    return [CT,CQ]




def _rescaleLinearly(targetVector,newLength):
    ''' Take a vector with an undersirable number of data points (e.g. twist/beta matrix) and rescales it to
    the desired dimension, assumes equally spaced data points.
    
    targetVector: Vector of points that need to be rescaled to the appropriate number
    newLength: Desired number of points
    '''
    
    assert(np.isscalar(newLength))
    
    
    x_old = np.linspace(0,1,len(targetVector))
    
    x_new = np.linspace(0,1,newLength)
    
    interp = scipy.interpolate.interp1d(x_old,targetVector,fill_value="extrapolate")
    
    return interp(x_new)
    




def HoverPerformance_Learned(b,R,C,omega,airfoil,rR,cR, twist,CT_nn,CQ_nn):
    ''' Main function used to compute CT and CQ based a learned heuristic for CT and CQ adjustment
    
    Params:
        b: Number of Blades (integer)
        
        R: Radius in [Meters]
        
        C: Chord [Meters] if varying chord please add maximum here and a discrete number of points in the cR variable
        
        omega: Rotation rate in [rad/s]
        
        r/R: (0,1] np.array with Position of blade elements as a function of Radius, needed when the chord is not constant
              
        c/R: Chord length divided by Radius, please make sure r/R and c/R refer to the same points
        
        twist: Either a vector with the twist at every blade element or the minimum and maximum if using linear twist. 
        Please Use DEGREES here, internally computations will be done in 
        
        airfoil: Functionality is still being added. Currently it is running with
        a Pandas dataframe [alpha,cl,cd] of your desired airfoil
        
        CT: A matrix of the form [w,b] numpy shape: (2,). Where the corrected CT will be CT_actual = w*CT_computed + b
        
        CQ: A matrix of the form [w,b] numpy shape: (2,). Where the corrected CQ will be CQ_actual = w*CQ_computed + b
    '''
    
    RHO = 1.225 #[kg/m3]
    
    assert(len(rR) == len(cR))
    
    assert(len(rR) == len(twist) or len(twist) == 2 or len(twist) == 0)
    
    
    # Tabulating elements if they were not provided
    if len(rR) == 0:
        rR,cR,twist = _tabulateBladeElements(C, R, twist)
        pass
    
    
    # Converting twist from Degrees to Radians
    
    twist = twist * (np.pi)/180
    
    
    
    # Obtaining theta from twist and zero angle of attack 
    '''ASSUMING ZERO ANGLE OF ATTACK COMPUTED ALREADY PLEASE ADD CODE HERE ONCE IT IS DONE '''
    
    alphaZL = 0
    
    theta = _getCollectivePitch(twist, alphaZL)
        
    
    
    
    # Note make sure this returns either a pandas series or a numpy ndarray
    a = _getLiftSlopeCurve(airfoil, omega, R, rR)
    
    
    
    # Step 5
    inducedAngle = _getInducedVelocity(a, b, cR, rR, theta)
    
    
    
    # Step 6
    
    alphas = _getLocalAOA(theta, inducedAngle)
    
    
    # Step 7 (Get cl and cd for every angle of attack on preceding list)
    
    
    '''refactor for efficiency and clarity '''
    
    aeroCoef_df = _extractClandCd(airfoil, alphas)
    
    cl = aeroCoef_df['CL'].to_numpy(dtype=float)
    
    cd = aeroCoef_df['CD'].to_numpy(dtype=float)
        
    # Step 8    
    
    runningThrustLoad = _getRunningThrustLoading(b,rR,cR,cl)
    
    
    # Step 9
    
    CTnoLoss = _getCTnoTipLoss(runningThrustLoad, rR)
    
    ## print("CT no loss: %s" % CTnoLoss)

    
    # Step 10
    
    B = _getTipLossFactor(CTnoLoss,b)
    
    
    ## print("B: %s " % B)
    
    # Step 11
    
    CT = _getCorrectedCT(CTnoLoss, runningThrustLoad, rR, B)
    
    ## print("CT: %s" % CT)

    
    
    # Step 12
    
    runProfile = _getRunningProfileTorqueCoefficient(b, rR, cR, cd)
    
    
    
    # Step 13
    
    CQ0 = _getProfileTorqueCoefficient(runProfile,rR)
    
    ## print("CQ0: %s" % CQ0)
    
    # Step 14
    
    runningInducedTorque = _getRunningInducedTorqueCoefficient(b,rR,cR,cl,inducedAngle)
    
    
    #Step 15
    
    CQi = _getInducedTorqueCoefficient(runningInducedTorque,rR)
    
    ## print("CQi: %s" % CQi)
    
    # Step 16
    
    
    delCQ = _computeDelCQ(CT,CQi)
    
    
    
    # Step 17
    
    diskLoading = _computeDiskLoading(CT,RHO,omega,R)
    
    
    # Step 18 
    
    ''' CALCULATED FOR A HEURISTIC'''
    
    CT_div_sigma = _computeRatioCTtoSolidity(CT,b,C,R)
    
    # Step 19
    
    ''' HEURISTIC '''
    
    wakeFactor = _computeWakeTorqueLinearCorrection(diskLoading,CT_div_sigma)
    
    
    # Step 20
    
    CQ = _computeTorqueCoefficient(CQ0,CQi,delCQ,wakeFactor)
    
    
    # Learned weights
    
    CT_LEARNED = CT_nn[0]*CT + CT_nn[1]
    
    CQ_LEARNEED = CQ_nn[0]*CQ + CQ_nn[1]
    
    
    ## print("CQ: %s" % CQ)    
    
    ## print("neu CT: %s" % CT_LEARNED)
    
    ## print("neu cq %s" % CQ_LEARNEED)
    
    ## print(CQ_nn)
    
    return np.array([CT_LEARNED,CQ_LEARNEED])




class UIUC_Propeller:
    def __init__(self,seed):
        ''' Seed should have the format [Prop_Name,Diameter,Pitch,geom_path,static_path]
        
        Params
        seed = [str,float,float,str(path),str(path)]    
        '''
        
        assert(os.path.exists(seed[3]))
        
        assert(os.path.exists(seed[4]))
        
        
        ''' Assuming all UIUC_Propellers have 2 blades only for now '''
        
        self.b = 2
        
        ''' above needs checking''' 
        
        self.NAME =  seed[0]
        
        self.Radius = seed[1]/2 # Note choosing to store radius instead of diameter for convinience
        
        self.pitch = seed[2]
        
        
        #self.twist = twist partitions, locations should match the rR elements
        
        #self.rR =  r/R blade element partitions
        
        #self.cR = non dimensionalized chord (c/R) partitions. Should match r/R
        
        self.rR,self.cR,self.twist = self._extractGeometricAttributes(seed[3])
        
        self.RPMs, self.CTs, self.CQs = self._extractStaticTestResults(seed[4])
        
        pass
    
    
    
    
    def _extractGeometricAttributes(self,geom_path):
        
        geodf = pd.read_csv(geom_path,delim_whitespace=True)
        
        # just in case one of the files has a typo.
        geodf.columns = ['r/R','c/R','beta']
        
        return [tuple(geodf['r/R']), tuple(geodf['c/R']), tuple(geodf['beta'])]
            
    
    
    
    def _extractStaticTestResults(self,static_path):
        
        statdf = pd.read_csv(static_path,delim_whitespace=True)
        
        statdf.columns = ['RPM','CT','CQ']
        
        return [tuple(statdf['RPM']), tuple(statdf['CT']), tuple(statdf['CQ'])]
    
    
    def _getMaxChord(self):
        
        return self.Radius * max(self.cR)
        
        pass
    
    
    
    
    def getTrainingData(self,blade_elements=15):
        ''' Return 2 matrices, the first one will contain all attributes of the propeller instance that called, 
        attributes will either be their in their standard form or one that is most convinient 
        (e.g. RPM will be converted to radians/sec, r/R and c/R will be rescaled to a fixed length, etc).
        This first matrix will be of the form:
        [b,Radius,Chord,RPM[in rad/s], r/R_rescaled,c/R_rescaled,twist_rescaled]
        
        In regression terms, this first matrix can be thought of as the 'predictors' or 'x'
        The second matrix will be:
        [CT,CQ]
        A single UIUC_Propeller object will thus correspond to several datapoints as each RPM and its associated CT,CQ
        will be included in separate rows.
        
        Parameters
        rescaled_blade_elements: The number of points in r/R,c/R,twist, these will be rescaled using '_rescaleLinearly'
                                    and will therefore utilize the same assumptions.
        
        Return:
            [x_mat,y_mat]
        '''
        
        C = self._getMaxChord()
        
        omega =  0.104719755 * np.array([self.RPMs])
        
        rR_rescaled = _rescaleLinearly(self.rR, blade_elements)
        
        cR_rescaled = _rescaleLinearly(self.cR, blade_elements)
        
        twist_rescaled = _rescaleLinearly(self.twist, blade_elements)
        
        
        
        
        # length column is based on : [b , R , C ,omega  ,rR[len = blade_elm],cr,twist] Thus: 4 + 3*rblade_element
        num_columns = blade_elements*3 + 4
        
        num_rows = len(self.RPMs)
        
        x_mat = np.zeros(shape=(num_rows,num_columns))
        
        y_mat = np.zeros(shape=(num_rows,2))
        

        
        for i, CURR_RPM in np.ndenumerate(omega):
            
            
                        
            y_mat[i[1],] = np.array([self.CTs[i[1]],self.CQs[i[1]]])

            
            x_mat[i[1],] = np.array([self.b,self.Radius, C ,CURR_RPM,*rR_rescaled,*cR_rescaled,*twist_rescaled])
            
            pass
        
        
        
        return [x_mat,y_mat]
    
    pass








#Testing    
# a = 2 * np.pi

# b = 2

# cR = 0.2

# rR = np.array([0.1,0.3,0.5,0.9])

# thetas = np.array([0.2,0.2,0.2,0.2])

naca4412 = pd.read_csv(os.path.join("Testing_Airfoils","NACA4412_RE500000.txt"),delim_whitespace=True)

# SC1095 = pd.read_csv("SC1095_RE500000.txt",delim_whitespace=True)


testProp = pd.read_csv(os.path.join("Testing_Propellers","apce_19x12_geom.txt"),delim_whitespace=True)

# # Scrapping unnecessary data

naca4412 = naca4412.iloc[1:,:3]

# SC1095 = SC1095.iloc[1:,:3]



# CT,CQ = HoverPerformance_Classical(2,0.48,0.2,100,naca4412,testProp['r/R'],testProp['c/R'],testProp['beta'])


# CT,CQ = HoverPerformance_Learned(2,0.48,0.2,100,naca4412,testProp['r/R'],testProp['c/R'],testProp['beta'],
#                                  np.array([1,0]),
#                                  np.array([1,0]))







# UH-60A Black hawk Data
# N blades: 4
# R = 26.83 ft = 8.18 meters
# chord = 1.75 ft = 0.5334
# Solidity 0.083
# Rotational speed 27 rad/s
# Nominal lift slope curve: 5.73 / rad
# linear twist = -18 degrees


# Note removed twist for r/R = 0 value, now twist is in degreees in increments of 0.05 r/R


# BLACKHAWK_TWIST = np.array([0,0,0,-0.25,-1.0,-1.75,-2.75,-3.5,-4.5,-5.25,-6.25,-7.25,-8,-8.75,-9.75,-10.25,-10.75,-12.25,-13,-11])

# test_UA60_twist = 18 + BLACKHAWK_TWIST

# UA60_twist = pd.DataFrame({'beta' : test_UA60_twist })

# blk_rR = np.linspace(0.05,1,20)

# blk_cR = np.full((20,),0.5334/8.18)

# dftest = pd.DataFrame({'r/R' : blk_rR,'c/R' : blk_cR, 'beta' : test_UA60_twist})


# CT,CQ = HoverPerformance_Classical(4,8.18,0.5334,27,SC1095,blk_rR,blk_cR,test_UA60_twist)


