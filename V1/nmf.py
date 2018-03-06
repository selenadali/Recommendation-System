# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:40:21 2018

@author: selen
"""

import numpy as np
from numpy import genfromtxt
from numpy import linalg as LA

#P: an initial matrix of dimension N x K, where is n is no of users and k is hidden latent features
#    P = np.random.rand(N,K)
    #Q : an initial matrix of dimension M x K, where M is no of movies and K is hidden latent features
#    Q = np.random.rand(M,K)
#steps : the maximum number of steps to perform the optimisation, hardcoding the values
#alpha : the learning rate, hardcoding the values
#beta  : the regularization parameter, hardcoding the values

# non negative regulaized matrix factorization implemention
def matrix_factorization(X,P,Q,K,steps,alpha,beta):
    Q = Q.T
    for step in range(steps):
        print(step)
        #for each user
        for i in range(X.shape[0]):
            #for each item
            for j in range(X.shape[1]):
                if X[i][j] > 0 :

                    #calculate the error of the element
                    eij = X[i][j] - np.dot(P[i,:],Q[:,j])
                    #second norm of P and Q for regularilization
                    sum_of_norms = 0
                    #for k in range(K):
                    #    sum_of_norms += LA.norm(P[:,k]) + LA.norm(Q[k,:])
                    #added regularized term to the error
                    sum_of_norms += LA.norm(P) + LA.norm(Q)
                    #print sum_of_norms
                    eij += ((beta/2) * sum_of_norms)
                    #print eij
                    #compute the gradient from the error
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * ( 2 * eij * Q[k][j] - (beta * P[i][k]))
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - (beta * Q[k][j]))

        #compute total error
        error = 0
        #for each user
        for i in range(X.shape[0]):
            #for each item
            for j in range(X.shape[1]):
                if X[i][j] > 0:
                    error += np.power(X[i][j] - np.dot(P[i,:],Q[:,j]),2)
        if error < 0.001:
            break
    return P, Q.T
