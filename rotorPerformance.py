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


  
# Step 4 choose collective pitch (if it is not given)

''' TODOOOOOOO CREATE '''

def getCollectivePitch(delTheta,alphaZL):
    
    
    return np.array([0])


''' '''
# Step 5 get inflow angle and induced velocity

# Function takes in lift slope curve a, number of blades b,
# r/R:  a vector of elements displaying the ratio of the blade element to the max radius
# c/R : a vector of elelements with the ratio of the local chord to the maximum radius
# theta: a vector of the local twist 

#returns a vector of the induced velocity of every point



def getInducedVelocity(a,b, cR, rR,theta):
    
    # check if cR and theta are either scalars or arrays of the appropriate length
    
    assert(np.isscalar(theta) or len(rR) == len(theta))
    
    assert(np.isscalar(cR) or len(cR) == len(rR))
    
    
    return  ((a * b * cR) / (16 *np.pi* rR)) * (-1 + np.sqrt(1 + (32 * np.pi * theta * rR) /(a * b * cR)  ) )
    

# Step 6


# theta and induced angle will be in RADIANS

# but the angle of attack returned will be in DEGREES due to historical convention for AOA

def getLocalAOA(theta, inducedAngle):
    
    return (theta - np.arctan(inducedAngle)) * (180/np.pi) 



''' TODO '''
# Step 7 -- extract cl,cd as a functon of mach number and airfoil Shape 









# Step 8 

# cl is assumed to be a numpy array of the appropriate length obtained in step 7

def getRunningThrustLoading(b,rR,cR,cl):
    
    assert(len(cl) == len(rR))
    
    
    return ((b * (rR**2) * cR * cl)/(2*np.pi))




# Step 9

# Integrate removing the assumed hub radius 

''' TODOOOO Change Integration to SPLINE '''

def getCTnoTipLoss(dct_dr, rR, x_hub = 0.1):
    
    # find index of every element larger than the hub
    
    assert(len(dct_dr) == len(rR))
    
    tck = scipy.interpolate.splrep(rR,dct_dr)
    
    
    
    
    
    
    # indexs = [i for i,l in enumerate(rR) if l > x_hub]
    
    # rR_toIntegrate = rR[indexs]
    
    # dct_dr_toIntegrate = dct_dr[indexs] 
    
    # scipy.integrate.simps(dct_dr_toIntegrate,rR_toIntegrate)
    
    return scipy.interpolate.splint(x_hub,1,tck)





# Step 10

'''TODOO make necessary changes to address the other two alternatived heuristics'''

def getTipLossFactor(CTnoTipLoss,b):
    
    assert(np.isscalar(CTnoTipLoss))

    assert(np.isscalar(b))
    
    assert(b > 0)
    
    return 1 - ((np.sqrt(2*CTnoTipLoss))/b)




# Step 11

''' TODO REFACTOR TO ACCOUNT FOR POSSSIBLE LACK OF POINTS '''

''' TODOOOO Change Integration to SPLINE '''


def getCorrectedCT(CTnoTipLoss, dct_dr, rR, B = 0.9):
    
    assert(len(dct_dr) == len(rR))
    
    # indexs = [i for i,l in enumerate(rR) if l > B]
    
    # rR_toIntegrate = rR[indexs]
    
    # dct_dr_toIntegrate = dct_dr[indexs] 
    
    # scipy.integrate.simps(dct_dr_toIntegrate,rR_toIntegrate)
    
    tck = scipy.interpolate.splrep(rR,dct_dr)
    
    
    return CTnoTipLoss - scipy.interpolate.splint(B,1,tck)

# step 12
    
def getRunningProfileTorqueCoefficient(b,rR, cR, cd):
    
    assert(len(rR) == len(cd))
    assert(len(rR) == len(cR))
    
    
    return (b * (rR**3) * cR * cd) / (2*np.pi)


# Step 13

''' TODOOOO Check which integration is more accurate when continuous func not used '''

    
def getProfileTorqueCoefficient(dq_dr,rR):
    
    assert(len(dq_dr) == len(rR))
    
    tck = scipy.interpolate.splrep(rR,dq_dr)
    
    return scipy.interpolate.splint(0,1,tck)


# Step 14

def getRunningInducedTorqueCoefficient(b,rR,cR,cl,induced_Vel):
    
    
    
    return (b *  (rR**3) * cR * cl * induced_Vel)/ (2*np.pi)





''' TODOOOO Change Integration to SPLINE '''

# Step 15
def getInducedTorqueCoefficient(dqi_dr,rR, x_hub = 0.1, B = 0.9):
    
    assert(len(dqi_dr) == len(rR))
    
    # indexs = [i for i,l in enumerate(rR) if l > B and l < x_hub]
    
    # rR_toIntegrate = rR[indexs]
    
    # dqi_dr_toIntegrate = dqi_dr[indexs] 
    
    # scipy.integrate.simps(dqi_dr_toIntegrate,rR_toIntegrate)
       
    tck = scipy.interpolate.splrep(rR,dqi_dr)
    
    return scipy.interpolate.splint(x_hub,B,tck)

# Step 16

'''TODO Address inherent problems in this heuristic 
        and make something thats better'''

def computeDelCQ(CT,CQi):
    
    assert(np.isscalar(CT))
    assert(np.isscalar(CQi))

    
    ct_candidates =  np.linspace(0,0.05,11)
    
    
    
    #From linear correction figure 1.29
    
    linearCorrection = np.array([0, 0.011, 0.028, 0.04, 0.053, 0.064, 0.075, 
                                0.083, 0.092, 0.10, 0.11])
    
    idx = np.abs(ct_candidates - CT).argmin()
    
    return linearCorrection[idx]*CQi


# Step 17

def computeDiskLoading(CT,rho,omega,R):
    
    return (CT * rho *( omega*R)**2)




# Step 18

def computeRatioCTtoSolidity(CT,b,c,R):
    
    return (CT)/(((b*c)/(np.pi*R)))




# Step 19

def computeWakeTorqueLinearCorrection(DiskLoading,CT_solidity):  
    # Source Figure 1.34
    
    '''TODO CREATE A CORRECTION THAT IS MORE ACCURATE'''
    
    return 0.94 + (1.06 * DiskLoading*CT_solidity)


# Step 20

def computeTorqueCoefficient(CQ,CQi,delCQ,wakeCorrection):
    
    assert(np.isscalar(CQ))
    
    assert(np.isscalar(CQi))

    assert(np.isscalar(delCQ))

    assert(np.isscalar(wakeCorrection))
    
    return (CQ + CQi + delCQ)*wakeCorrection




a = 2*np.pi

b = 2

cR = 0.2

rR = np.array([0.1,0.3,0.5,0.9])

thits = np.array([0.2,0.2,0.2,0.2])
