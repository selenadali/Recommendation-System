# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mlt
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
#import bokeh.plotting as bp
#from bokeh.plotting import save
from bokeh.models import HoverTool
from bokeh.plotting import figure, show, ColumnDataSource
from sklearn import cluster
#import matplotlib.cm as cm
import pandas as pd
import csv
from matplotlib.backends.backend_pdf import PdfPages
    
def split_data(ratings): # 90% training, 10% test
    n = len(ratings)
    n_train = round(9/10*n)
    ratings_random = ratings.copy()
    np.random.shuffle(ratings_random)
    data_train = ratings_random[:n_train]
    data_test = ratings_random[n_train:]
    with open("data_train.csv", "w", newline="") as csv_file:
            csv_file.write("userId,movieId,rating,timestamp\n")
            writer = csv.writer(csv_file, delimiter=',')
            for line in data_train:
                writer.writerow(line)              
    with open("data_test.csv", "w", newline="") as csv_file:
            csv_file.write("userId,movieId,rating,timestamp\n")
            writer = csv.writer(csv_file, delimiter=',')
            for line in data_test:
                writer.writerow(line)           
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
    plt.savefig("tsne.pdf")
    
    
    tsne_represent(Nu, nb_iter, Nu_embedded, ratings,data_train)
        
def tsne_represent(Nu, nb_iter, Nu_embedded, ratings,data_train):
    
#    tsne = TSNE(init='pca', perplexity=40, learning_rate=1000, n_components=2,
#            early_exaggeration=8.0, n_iter = nb_iter, random_state=0, metric='l2')
#    tsne_representation = tsne.fit_transform(Nu)
#        
    cl = cluster.AgglomerativeClustering(10)
#    cl.fit(tsne_representation)
    cl.fit(Nu_embedded)
    tsne_representation = Nu_embedded
   
#    ratings_df = pd.DataFrame(ratings[1:], columns = ['userId', 'movieId', 'rating'], dtype = int)
#    y = ratings_df.pop('rating')

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    cmap = plt.cm.get_cmap('jet')
   
#    m = cm.ScalarMappable(cmap=cm.jet)
#    m.set_array(ratings_df.pop('rating'))
    
    plt.scatter(tsne_representation[:,0], tsne_representation[:,1],alpha=0.5,  cmap=cmap, s=20)
#    plt.colorbar()
    plt.savefig("tsne_rating.pdf")

    plt.subplot(1,2,2)
    plt.scatter(tsne_representation[:,0], tsne_representation[:,1],c=cl.labels_, marker='s', s=20)
    plt.savefig("tsne_cluster.pdf")
    
    tsne_interactive(tsne_representation)
    
def tsne_interactive(tsne_representation):
    
    
    df_train = pd.read_csv("data_train.csv")  # load train
    df_test = pd.read_csv("data_test.csv")    # load test
    
    print("x")
    
    y = df_train.pop('rating')        
    df = pd.concat([df_train, df_test], axis=0).reset_index()  # train + test data

    source_train = ColumnDataSource(
        data=dict(
            x = tsne_representation[:len(y),0],
            y = tsne_representation[:len(y),1],
            desc = y,
            colors = ["#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in 
                      255*mlt.cm.jet(mlt.colors.Normalize()(y.values))],
            userId = df["userId"].iloc[:len(y)],
            movieId = df["movieId"].iloc[:len(y)],
            rating = df["rating"].iloc[:len(y)]
        )
    )
    
    
    print("x")
    
    source_test = ColumnDataSource(
            data=dict(
                x = tsne_representation[len(y):,0],
                y = tsne_representation[len(y):,1],
                userId = df["userId"].iloc[:len(y)],
                movieId = df["movieId"].iloc[:len(y)],
                rating = df["rating"].iloc[:len(y)],
            )
        )
    
    hover_tsne = HoverTool(names=["test", "train"], tooltips=[("rating", "@desc"), 
                                     ("userId", "@userId"), 
                                     ("movieId", "@movieId"), 
                                     ("rating", "@rating")])
    tools_tsne = [hover_tsne, 'pan', 'wheel_zoom', 'reset']
    
    print("x")
    
    plot_tsne = figure(plot_width=600, plot_height=600, tools=tools_tsne, title='rating')
    
    plot_tsne.square('x', 'y', size=7, fill_color='orange', 
                     alpha=0.9, line_width=0, source=source_test, name="test")
    plot_tsne.circle('x', 'y', size=10, fill_color='colors', 
                     alpha=0.5, line_width=0, source=source_train, name="train")
    
    show(plot_tsne)
      
      
test1()

