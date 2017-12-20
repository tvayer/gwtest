#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 09:15:28 2017

@author: nico
"""

import numpy as np

import ot,time

class WGW():
    def __init__(self,G,C1,C2,p,q,epsilon,alpha,loss_fun='square_loss'):
        self.loss_fun=loss_fun
        self.C1 = np.asarray(C1, dtype=np.float64)
        self.C2 = np.asarray(C2, dtype=np.float64)
        self.T = np.eye(len(p), len(q))
        self.alpha=alpha
        self.epsilon=epsilon
        f1,f2,h1,h2=self.create_tens_func()
        constC1=np.dot(np.dot(f1(C1),p.reshape(-1,1)),np.ones(len(q)).reshape(1,-1))
        constC2=np.dot(np.ones(len(p)).reshape(-1,1),np.dot(q.reshape(1,-1),f2(C2).T))
        self.constC=constC1+constC2
        self.hC1 = h1(C1)
        self.hC2 = h2(C2)
        self.G=G
        self.p=p
        self.q=q

    def create_tens_func(self):

        if self.loss_fun == 'square_loss':
            def f1(a):
                return (a**2) 
            def f2(b):
                return (b**2)    
            def h1(a):
                return a    
            def h2(b):
                return 2*b
        elif loss_fun == 'kl_loss':
            def f1(a):
                return a * np.log(a + 1e-15) - a    
            def f2(b):
                return b    
            def h1(a):
                return a    
            def h2(b):
                return np.log(b + 1e-15)

        return f1,f2,h1,h2

    def wgw(self,max_iter=1000, tol=1e-9, verbose=False, log=False):

        cpt = 0
        err = 1

            
        log_struct={}
        log_struct['err']=[]
        log_struct['GW_dist']=[]
        log_struct['sinkhorn']=[]
        log_struct['cpt']=0

        while (err > tol and cpt < max_iter):

            tens = self.constC-np.dot(self.hC1, self.T).dot(self.hC2.T)
            Cost = self.G+self.alpha*tens

            start=time.time()        
            self.T=ot.sinkhorn(self.p, self.q, Cost, self.epsilon, numItermax=100)
            end=time.time()

            log_struct['sinkhorn'].append(end-start)

            log_struct['GW_dist'].append(np.sum(self.T*Cost))
            if cpt>1:
                err = (log_struct['GW_dist'][-1]-log_struct['GW_dist'][-2])**2
            
                if log:
                    log_struct['err'].append(err)
                    log_struct['cpt']=cpt

                if verbose:
                    print('{:5d}|{:8e}|'.format(cpt, err))

            cpt += 1


        if log:
            return self.T, log_struct
        else:
            return self.T



