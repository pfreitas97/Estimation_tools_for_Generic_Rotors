#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 19:30:26 2020

@author: pedroaugustofreitasdearaujo
"""

import unittest

import rotorPerformance

import numpy as np

import scipy.integrate



class Test_hoverRotor(unittest.TestCase):
    
    
    # Step 5
    
    def test_getInducedVellocity(self):
        """
        Test correctnness of the induced angle found [radians]
        """
        a_test = (np.pi**2)/90 # theoretical value for thin airfoil
        
        b_test = 2
        
        cR_test = np.array([0.9,0.1,0.2])
        
        rR_test = np.array([0.1,0.5,0.9])
        
        theta_test = 0.1 # In radians so about 5.7 degrees
    
        CORRECT = np.array([0.0576636,0.0123672,0.0129898013])
                        
        result = rotorPerformance._getInducedVelocity(a_test, b_test, cR_test, rR_test, theta_test)
        
        
        self.assertTrue(all(np.round(result,3) == np.round(CORRECT,3)))
        pass
    
    
    
    # Step 9 Integration test
    
    def test_getCTnoTipLoss(self):
        '''
        Test the accuracy of numerical integration method
        '''
        
        rR = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]) # Using 5 discrete points evenly spaced
        
        x_squared = lambda x : x**2
    
        x_squared_defINTEGRAL = lambda x_0,x_1 : (1/3)*(x_1**3) - (1/3)*(x_0**3)
        
        x2_integrand = np.array(x_squared(rR))
        
        
        X_HUB = 0.1
        
        #print(x2_integrand)
        
        x2_test = rotorPerformance._getCTnoTipLoss(x2_integrand,rR,X_HUB)
        
        # print(x2_test)
        
        # print(x_squared_defINTEGRAL(X_HUB,1))
        
        # print(scipy.integrate.trapz(x2_integrand,rR))
        
        # ftest = scipy.interpolate.interp1d(rR,x2_integrand,fill_value="extrapolate")
        
        # print(scipy.integrate.quad(ftest,0,1))
        
        
        self.assertTrue( (x2_test - x_squared_defINTEGRAL(X_HUB,1))**2 < 0.1 )
            
        
        pass
    
    
    
    
        
        
        
        

if __name__ == '__main__':
    unittest.main()