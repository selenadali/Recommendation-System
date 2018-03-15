# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pylab
import csv
import codecs
import re

import matplotlib as mlt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#from bokeh.plotting import save
from bokeh.models import HoverTool
from bokeh.plotting import figure, show, output_notebook, save
from bokeh.io import curdoc
from bokeh.layouts import widgetbox, row, column
from bokeh.models import ColumnDataSource, Select, Slider
from bokeh.palettes import Spectral6

from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from scipy.cluster.hierarchy import dendrogram

def new_data():
    unames = ['user_id','gender','age','occupation','zip']
    users = pd.read_table('users.dat',sep='::',header=None, names=unames)
    rnames = ['user_id','movie_id','rating','timestamp']    
    ratings = pd.read_table('ratings.dat',sep='::',header=None,names=rnames)
    mnames = ['movie_id','title','genres']
    movies = pd.read_table('movies.dat',sep='::',header=None,names=mnames)
    data = pd.merge(pd.merge(ratings,users),movies)
    #ratinngs by gender for each movie
    mean_ratings = data.pivot_table('rating',index= 'title', columns='gender',aggfunc='mean')
    ratings_by_title = data.groupby('title').size()
    active_titles = ratings_by_title.index[ratings_by_title >= 250]
    mean_ratings = mean_ratings.ix[active_titles]
    top_female_ratings = mean_ratings.sort_index(by='F',ascending=False)
    top_male_ratings = mean_ratings.sort_index(by='M',ascending=False)
    mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
    sorted_by_diff = mean_ratings.sort_index(by='diff')
    
    rating_std_by_title = data.groupby('title')['rating'].std()
    rating_std_by_title = rating_std_by_title.ix[active_titles]
    rating_std_by_title.sort_index(ascending=False)[:10]
    
    avg_age = data.groupby('title')['age'].mean()
    avg_age = avg_age.ix[active_titles]
    avg_age.sort_index(ascending=True)[:10]
    
    return users, movies, ratings, data
    

def preprocessing(row):
    line = ','.join(row)
    line1 = line.split("\"")
    #re.sub
    name_with_comma = re.findall(r'"([^"]*)"', line)
    if name_with_comma != []:
        name_with_comma = ' '.join(''.join(name_with_comma).split(","))
        line1[1] = name_with_comma
        new_line = " ".join(line1)
        
        return new_line
    else:
        return line 

def get_movies():
    lines = list()
    #encoding="utf8"
    with open('movies_new.csv',encoding="utf8") as f:
      reader = csv.reader(f)
      for row in reader:
          lines.append(preprocessing(row))
                
    with open("movies_cleaned_new.csv", "w", encoding="utf8", newline='') as csv_file:
#            csv_file.write("movieId,title,genres\n")
            writer = csv.writer(csv_file)
            for line in lines:
                writer.writerow([line])
                
    #movies.csv has non-numeric values which need encoding
    types_of_encoding = ["utf8", "cp1252"]
    for encoding_type in types_of_encoding:
        with codecs.open('movies_cleaned_new.csv', encoding = encoding_type, errors ='replace') as f:
            movies = [tuple(line) for line in csv.reader(f)]
            
    for i in range(len(movies)):
        movies[i] = movies[i][0].split("::")
    return movies
    
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
#    ratings = np.genfromtxt('ratings.csv', delimiter=",", dtype=(int,int,float,int))
#    ratings = ratings[1:]
#    data_train,data_test = split_data(ratings)
    
    users, movies, ratings, data = new_data()
    ratings = np.genfromtxt('ratings.dat', delimiter="::", dtype=(int,int,int,int))

    data_train,data_test = split_data(ratings)

    
    nb_users = 6040
    nb_movies = 3952
    z = 3
    eps = 5e-3 #10e-3,5e-3
    epsu = 10e-5
    epsi = 10e-5
    nb_iter = 10000
    nb_norm = 100
    Nu,Ni = init_matrix(nb_users,nb_movies,z,5,0)
    iter_histo,error_histo,Nu,Ni = SGD(data_train,data_test,Nu,Ni,nb_iter,nb_norm,eps,epsu,epsi)
    #plt.figure()
    #plt.plot(iter_histo,error_histo)
    #plt.show()
    Nu_embedded = TSNE(n_components=2).fit_transform(Nu)
    #plt.figure()
    #plt.plot(Nu_embedded[:,0],Nu_embedded[:,1], 'b*')
    #plt.savefig("tsne.pdf")
    Ni = np.resize(Ni,(z,nb_users))
    Ni_embedded = TSNE(n_components=2).fit_transform(Ni.transpose())
    
    tsne_representation, c_u, c_i = tsne_represent(Nu, nb_iter, Nu_embedded, ratings,data_train, Ni, Ni_embedded)
    source_u = tsne_users(tsne_representation,c_u)   
    source_i = tsne_movies(Nu, nb_iter, Nu_embedded, ratings,data_train, Ni, nb_movies,Ni_embedded,c_i)

    
def tsne_represent(Nu, nb_iter, Nu_embedded, ratings,data_train, Ni, Ni_embedded):
    cl = cluster.AgglomerativeClustering(10)
    cl.fit(Nu_embedded)
    c_u=cl.labels_
    tsne_representation = Nu_embedded
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    cmap = plt.cm.get_cmap('jet')    
    plt.scatter(tsne_representation[:,0], tsne_representation[:,1],alpha=0.5,  cmap=cmap, s=20)
    plt.savefig("tsne_rating.pdf")
    plt.subplot(1,2,2)
    plt.scatter(tsne_representation[:,0], tsne_representation[:,1],c=c_u, marker='s', s=20)
    plt.savefig("tsne_cluster.pdf")
    
    cl2 = cluster.AgglomerativeClustering(10)
    cl2.fit(Ni_embedded)
    c_i=cl2.labels_
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    cmap = plt.cm.get_cmap('jet')    
    plt.scatter(Ni_embedded[:,0], Ni_embedded[:,1],alpha=0.5,  cmap=cmap, s=20)
    plt.savefig("tsne_film_rating.pdf")
    plt.subplot(1,2,2)
    plt.scatter(Ni_embedded[:,0], Ni_embedded[:,1],c=c_i, marker='s', s=20)
    plt.savefig("tsne_film_cluster.pdf")
    
    return tsne_representation, c_u, c_i
 
def tsne_users(tsne_representation,c):    
    df_train = pd.read_csv("data_train.csv")  # load train
    df_test = pd.read_csv("data_test.csv")    # load test
    movies = get_movies()

    df_movies = pd.DataFrame(movies[1:], columns = ['movieId', 'title', 'genres'])

    y = df_train['rating']        
    df = pd.concat([df_train, df_test], axis=0).reset_index()  # train + test data

#    dfc = df_movies["genres"]
#    dfc2 = pd.concat([dfc, df_movies], axis=1)
#    df3 = pd.concat([dfc, df], axis=1)
#        
    palette ={1:'red',2:'green',3:'blue',
              4:'yellow',5:"orange", 6:"pink", 
              7:"black", 8:"purple", 9:"brown",0:"white"}
    colors =[]
    for i in c:
            colors.append(palette[i])    
            
    source_train = ColumnDataSource(
        data=dict(
            x = tsne_representation[:len(y),0],
            y = tsne_representation[:len(y),1],
            desc = y,
            colors = colors,
            userId = df_train["userId"].iloc[:len(y)],
            movieId = df_train["movieId"].iloc[:len(y)],
            rating = df_train["rating"].iloc[:len(y)],
#            genre = df3["genres"].iloc[:len(y)]
        )
    )
    y = df_test['rating']        

    source_test = ColumnDataSource(
            data=dict(
                x = tsne_representation[:len(y),0],
                y = tsne_representation[:len(y),1],
                userId = df["userId"].iloc[:len(y)],
                movieId = df["movieId"].iloc[:len(y)],
                rating = df["rating"].iloc[:len(y)],
#                genre = df_movies["genres"].iloc[:len(y)],
            )
        )
            
    hover_tsne = HoverTool(tooltips=[("rating", "@desc"), 
                                     ("userId", "@userId"), 
                                     ("movieId", "@movieId"), 
                                     ("rating", "@rating"),
                                     ]) #("genre", "@genres")
    tools_tsne = [hover_tsne, 'pan', 'wheel_zoom', 'reset']
    
    plot_tsne = figure(plot_width=600, plot_height=600, tools=tools_tsne, title='rating')
    
    plot_tsne.circle('x', 'y', size=10, fill_color='colors', 
                     alpha=0.5, line_width=0, source=source_train, name="bokeh")
    
#    hover = plot_tsne.select(dict(type=HoverTool))
#    hover.tooltips = {"content": "rating: @desc, userId: @userId, movieId: @movieId, rating = @rating"}

    show(plot_tsne)
    return source_train
    
  
      
def tsne_movies(Nu, nb_iter, Nu_embedded, ratings,data_train, Ni, nb_movies, Ni_embedded,c_i):

    title = 'T-SNE visualization of embedding'    
    movies = get_movies()
    movies = pd.DataFrame(movies[1:], columns = ['movieId', 'title', 'genres'])    
    ratings_pd = pd.read_csv('ratings.csv') #loading ratings file for groupby    
    movie_names = movies.set_index('movieId')['title'].to_dict() #creating dictionary of movieid: movie title
    movie_genres = movies.set_index('movieId')['genres'].to_dict() #creating dictionary of movieid: movie title
    g=ratings_pd.groupby('movieId')['rating'].count() #counting the number of ratings for each movie
    topMovies=g.sort_values(ascending=False).index.values[:671] #top 671 movies based on number of ratings    
    dicts = [movie_names,movie_genres]
    super_dict = {}
    for d in dicts:
        for k, v in d.items():  # d.items() in Python 3+
            super_dict.setdefault(k, []).append(v)
            super_dict.keys

    #visualizing t-sne components of embeddings using Bokeh
    df_combine = pd.DataFrame([super_dict.get(str(i)) for i in topMovies])
    df_combine.columns = ['title','genre']
    df_combine['x-tsne'] = Ni_embedded[:,0]
    df_combine['y-tsne'] = Ni_embedded[:,1]
    
    palette ={1:'red',2:'green',3:'blue',
              4:'yellow',5:"orange", 6:"pink", 
              7:"black", 8:"purple", 9:"brown",0:"white"}
    colors =[]
    for i in c_i:
            colors.append(palette[i])  
    
    print(colors)
            
    source = ColumnDataSource(dict(
        x=df_combine['x-tsne'],
        y=df_combine['y-tsne'],
        title= df_combine['title'],
        genre= df_combine['genre'],
        colors = colors
        ))
    
    hover_tsne = HoverTool( tooltips=[ 
                                     ("Title", "@title"), 
                                     ("Genre", "@genre")
                                     ]) 
    tools_tsne = [hover_tsne, 'pan', 'wheel_zoom', 'reset']
    
    plot_tsne = figure(plot_width=600, plot_height=600, tools=tools_tsne, title='rating')
    
    plot_tsne.circle('x', 'y', size=10, fill_color='colors', 
                     alpha=0.5, line_width=0, source=source, name="bokeh")
    
    # hover tools
    hover = plot_tsne.select(dict(type=HoverTool))
    hover.tooltips = {"content": "Title: @title, Genre: @genre"}

    show(plot_tsne)
    
    return source 


    
test1()



