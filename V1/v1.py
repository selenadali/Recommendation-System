# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 11:36:53 2018

@author: selen
"""

import pandas as pd
import numpy as np
import csv
import codecs
from scipy.sparse.linalg import svds
import nimfa
from sklearn.preprocessing import normalize
import nmf

def main():
    
    

    #import excel files as tuple 
    ratings_file = "ratings.csv"
    with open(ratings_file) as f:
        ratings=[tuple(line[:-1]) for line in csv.reader(f)]
    
    #movies.csv has non-numeric values which need encoding
    types_of_encoding = ["utf8", "cp1252"]
    for encoding_type in types_of_encoding:
        with codecs.open('movies.csv', encoding = encoding_type, errors ='replace') as f:
            movies = [tuple(line) for line in csv.reader(f)]
        
    #print(ratings)
#    print(movies)
    
    #Simplify the view with dataframe, I am not going to use for manipulations it is just create matrix
    ratings_df = pd.DataFrame(ratings[1:], columns = ['userId', 'movieId', 'rating'], dtype = int)
    movies_df = pd.DataFrame(movies[1:], columns = ['movieId', 'title', 'genres'])
    movies_df['movieId'] = movies_df['movieId'].apply(pd.to_numeric)
    
#    print(ratings_df)
#    print(movies_df)

    #create matrix movieIds are in columns, userIds are in lines
    R_df = ratings_df.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
    R_df.head()
    
#    print(R_df)
    
    #corvert to numpy
    R = R_df.as_matrix()
    
    R = R.astype(float)
#    print(R)
    
    #normalisation
    user_ratings_mean = np.mean(R, axis = 1, dtype = float)
    print(user_ratings_mean)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)
    print("R_demeaned",R_demeaned)
    
    normed_matrix = normalize(R, axis=1, norm='l2')
    print("normed_matrix",normed_matrix)

    #singular value decomposition (from scipy.sparse.linalg import svds)
    U, sigma, Vt = svds(R_demeaned, k = 50)
    
    #Σ  returned is just the values instead of a diagonal matrix. 
    #This is useful, but since I’m going to leverage matrix multiplication to get predictions 
    #I’ll convert it to the diagonal matrix form.
    
    sigma = np.diag(sigma)
    
    #Here we arrive the idea of NMF, SVD can have negative values which makes interpretation more difficult
    #We have V = UΣVt where U and Vt have mixed values, we need to pass V = WHt with all nonnegative values.
    #Objective min ||V - WH||^2
    
    #Nmf(R, seed="random_vcol", rank=2, max_iter = 100)
    #nmf = nimfa.Nmf(R)
    #fit = nmf()
    #W = fit.basis()
    #H = fit.coef()
    #
    #print(W)
    #print(H)
    
    W, H = factorize(normed_matrix)
#    print(W)
#    print(H)
    
#    steps : the maximum number of steps to perform the optimisation was set to 5000
#   alpha : the learning rate was set to 0.0002
#   beta : the regularization parameter was set to 0.02
#   k : hidden latent features was set to 8
    K=8
    step=5
    alpha=0.0002
    beta=0.02
    #no of users
    N= R.shape[0]
    #no of movies
    M = R.shape[1]
    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)
    
    P, Q = nmf.matrix_factorization(R,P,Q,K,step,alpha,beta)
    print("P",P)
    print("QT",Q)
#:param V: The MovieLens data matrix. 
#    :type V: `numpy.matrix`
def factorize(V):
    
    snmf = nimfa.Snmf(V, seed="random_vcol", rank=3, max_iter=3, version='r', eta=1.,
                      beta=1e-4, i_conv=10, w_min_change=0)
    print("Algorithm: %s\nInitialization: %s\nRank: %d" % (snmf, snmf.seed, snmf.rank))
    fit = snmf()
    sparse_w, sparse_h = fit.fit.sparseness()
    print("""Stats:
            - iterations: %d
            - Euclidean distance: %5.3f
            - Sparseness basis: %5.3f, mixture: %5.3f""" % (fit.fit.n_iter,
                                                            fit.distance(metric='euclidean'),
                                                            sparse_w, sparse_h))
    return fit.basis(), fit.coef()
    
    
    
    
    
    
    
    
main()
    


