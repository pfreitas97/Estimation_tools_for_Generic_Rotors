#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 19:30:26 2020

@author: pedroaugustofreitasdearaujo
"""

import unittest

import rotorPerformance

import numpy as np



class Test_hoverRotor(unittest.TestCase):
    
    
    def test_getInducedVellocity(self):
        """
        Test that it can sum a list of integers
        """
        a_test = 2*np.pi
        
        b_test = 2
        
        cR_test = np.array([0.1,0.1,0.1,0.2,0.2])

        rR_test = np.array([0.1,0.3,0.5,0.7,1])

        theta_test = 10  
        
        
        
        
        
        
        # ((a * b * cR) / (16 *np.pi* rR)) * (-1 + np.sqrt(1 + (32 * np.pi * theta * rR) /(a * b * cR)  ) )
        
        
        
        
        result = rotorPerformance.getInducedVelocity(a_test, b_test, cR_test, rR_test, theta_test)
        
        
        
        
        
        self.assertEqual(result, 6)
        
        
        
        
        

if __name__ == '__main__':
    unittest.main()