# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def split_data(ratings): # 90% training, 10% test
    n = len(ratings)
    n_train = round(9/10*n)
    ratings_random = ratings.copy()
    np.random.shuffle(ratings_random)
    data_train = ratings_random[:n_train]
    data_test = ratings_random[n_train:]
    return data_train,data_test


def init_matrix(nb_users,nb_movies,z,a,b):
    # a et b définis l'horizon de tirage a*(0,1) -b
    Nu = a*np.random.random((nb_users,z)) - b
    Ni = a*np.random.random((z,nb_movies)) - b
    Nu /= Nu.sum(axis = 0)[np.newaxis,:]
    return Nu, Ni


def calcul_error(Nu,Ni,data_test):
    error = 0
    for e in data_test:
        (idu,idi,rating,time) = e
        error += (np.dot(Nu[idu-1,:],Ni[:,idi-1]) - rating)**2
        # la partie régularization
    return error     
        
def SGD(data_train,data_test,Nu,Ni,nb_iter,nb_norm,eps,epsu,epsi): # descente de gradiant stochastique
    nb = len(data_train)
    j = nb_norm
    error_histo = []
    iter_histo = []
    for i in range(nb_iter):
        (idu,idi,rating,time) = data_train[np.random.randint(0,nb)]
        du = 2*(np.dot(Nu[idu-1,:],Ni[:,idi-1]) - rating)*Ni[:,idi-1]
        di = 2*(np.dot(Nu[idu-1,:],Ni[:,idi-1]) - rating)*Nu[idu-1,:]
        Nu[idu-1,:] -= eps*du
        Ni[:,idi-1] -= eps*di
        for k in range(Nu.shape[1]):
            Nu[idu-1,k] = max(0,Nu[idu-1,k])
            Ni[k,idi-1] = max(0,Ni[k,idi-1])
        j -= 1
        if j <= 0 :
            j = nb_norm
            Nu *= 1-epsu
            Ni *= 1-epsi
            iter_histo.append(i)
            error_histo.append(calcul_error(Nu,Ni,data_test))
        #Nu /= Nu.sum(axis = 0)[np.newaxis,:]
        #Ni /= Ni.sum(axis = 1)[:,np.newaxis]
    return iter_histo,error_histo,Nu,Ni


def test1():
    ratings = np.genfromtxt('ratings.csv', delimiter=",", dtype=(int,int,float,int))
    ratings = ratings[1:]
    data_train,data_test = split_data(ratings)
    nb_users = 671
    nb_movies = 164979
    z = 3
    eps = 5e-3 #10e-3,5e-3
    epsu = 10e-5
    epsi = 10e-5
    nb_iter = 10000
    nb_norm = 100
    Nu,Ni = init_matrix(nb_users,nb_movies,z,5,0)
    iter_histo,error_histo,Nu,Ni = SGD(data_train,data_test,Nu,Ni,nb_iter,nb_norm,eps,epsu,epsi)
    plt.figure()
    plt.plot(iter_histo,error_histo)
    plt.show()
    Nu_embedded = TSNE(n_components=2).fit_transform(Nu)
    plt.figure()
    plt.plot(Nu_embedded[:,0],Nu_embedded[:,1], 'b*')
    