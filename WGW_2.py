#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 09:15:28 2017

@author: nico
"""

import numpy as np

import ot


def wgw(G, C1, C2, p, q, loss_fun, epsilon,alpha,
                       max_iter=1000, tol=1e-9, verbose=False, log=False):
    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)

    T = np.outer(p, q)  # Initialization

    cpt = 0
    err = 1

    if loss_fun == 'square_loss':
        def f1(a):
            return (a**2) / 2
        def f2(b):
            return (b**2) / 2    
        def h1(a):
            return a    
        def h2(b):
            return b
    elif loss_fun == 'kl_loss':
        def f1(a):
            return a * np.log(a + 1e-15) - a    
        def f2(b):
            return b    
        def h1(a):
            return a    
        def h2(b):
            return np.log(b + 1e-15)
        
    constC = np.dot(f1(C1),p).reshape(len(p),1)*np.ones(len(q))+np.dot(f2(C2),q).reshape(len(q),1)*np.ones(len(p))
    hC1 = h1(C1)
    hC2 = h2(C2)
        
    log_struct={}
    log_struct['err']=[]
    log_struct['GW_dist']=[]
    while (err > tol and cpt < max_iter):
        tens = constC-np.dot(hC1, T).dot(hC2.T)
        Cost = G+alpha*tens        
        T = ot.sinkhorn(p, q, Cost, epsilon, numItermax=100)

        log_struct['GW_dist'].append(np.sum(T*Cost))
        if cpt>1:
            err = (log_struct['GW_dist'][-1]-log_struct['GW_dist'][-2])**2
        
            if log:
                log_struct['err'].append(err)

            if verbose:
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    if log:
        return T, log_struct
    else:
        return T