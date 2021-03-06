# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mlt
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
#import bokeh.plotting as bp
#from bokeh.plotting import save
from bokeh.models import HoverTool
from bokeh.plotting import figure, show, ColumnDataSource, output_notebook, save
from sklearn import cluster
#import matplotlib.cm as cm
import pandas as pd
import csv
from matplotlib.backends.backend_pdf import PdfPages
import codecs
import re

def preprocessing(row):
    line = ','.join(row)
    line1 = line.split("\"")
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
    with open('movies.csv', encoding="utf8") as f:
      reader = csv.reader(f)
      for row in reader:
          lines.append(preprocessing(row))
                
    with open("movies_cleaned.csv", "w", encoding="utf8", newline='') as csv_file:
#            csv_file.write("movieId,title,genres\n")
            writer = csv.writer(csv_file)
            for line in lines:
                writer.writerow([line])
                
    #movies.csv has non-numeric values which need encoding
    types_of_encoding = ["utf8", "cp1252"]
    for encoding_type in types_of_encoding:
        with codecs.open('movies_cleaned.csv', encoding = encoding_type, errors ='replace') as f:
            movies = [tuple(line) for line in csv.reader(f)]
            
    for i in range(len(movies)):
        movies[i] = movies[i][0].split(",")
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
    
    
    tsne_represent(Nu, nb_iter, Nu_embedded, ratings,data_train, Ni)
    tsne_movies(Nu, nb_iter, Nu_embedded, ratings,data_train, Ni, nb_movies)
    
def tsne_represent(Nu, nb_iter, Nu_embedded, ratings,data_train, Ni):
    
#    tsne = TSNE(init='pca', perplexity=40, learning_rate=1000, n_components=2,
#            early_exaggeration=8.0, n_iter = nb_iter, random_state=0, metric='l2')

#Perplexity is a metric for how many neighbors a point has,
# and significantly affects the algorithm’s output:
#    tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0,learning_rate=200.0, n_iter=nb_iter, n_iter_without_progress=300,min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0,random_state=None, method='barnes_hut', angle=0.5)
#    tsene_representation = tsne.fit_transform(Ni.transpose())
#        
    cl = cluster.AgglomerativeClustering(10)
#    cl.fit(tsne_representation)
    cl.fit(Nu_embedded)
    tsne_representation = Nu_embedded
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    cmap = plt.cm.get_cmap('jet')    
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
    movies = get_movies()

    df_movies = pd.DataFrame(movies[1:], columns = ['movieId', 'title', 'genres'])

    y = df_train['rating']        
    df = pd.concat([df_train, df_test], axis=0).reset_index()  # train + test data

#    dfc = df_movies["genres"]
#    dfc2 = pd.concat([dfc, df_movies], axis=1)
#    df3 = pd.concat([dfc, df], axis=1)
#    
    source_train = ColumnDataSource(
        data=dict(
            x = tsne_representation[:len(y),0],
            y = tsne_representation[:len(y),1],
            desc = y,
            colors = ["#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in 
                      255*mlt.cm.jet(mlt.colors.Normalize()(y.values))],
            userId = df_train["userId"].iloc[:len(y)],
            movieId = df_train["movieId"].iloc[:len(y)],
            rating = df_train["rating"].iloc[:len(y)],
#            genre = df3["genres"].iloc[:len(y)]
        )
    )
    
    
    print("x")
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
    
    hover_tsne = HoverTool(names=["test", "train"], tooltips=[("rating", "@desc"), 
                                     ("userId", "@userId"), 
                                     ("movieId", "@movieId"), 
                                     ("rating", "@rating")
                                     ]) #("genre", "@genres")
    tools_tsne = [hover_tsne, 'pan', 'wheel_zoom', 'reset']
    
    print("x")
    
    plot_tsne = figure(plot_width=600, plot_height=600, tools=tools_tsne, title='rating')
    
    plot_tsne.square('x', 'y', size=7, fill_color='orange', 
                     alpha=0.9, line_width=0, source=source_test, name="test")
    # show(plot_tsne)
    
    plot_tsne.circle('x', 'y', size=10, fill_color='colors', 
                     alpha=0.5, line_width=0, source=source_train, name="train")
    
    show(plot_tsne)
    
    
      
def tsne_movies(Nu, nb_iter, Nu_embedded, ratings,data_train, Ni, nb_movies):
    
#    tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0,learning_rate=200.0, n_iter=nb_iter, n_iter_without_progress=300,min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0,random_state=None, method='barnes_hut', angle=0.5)
    Ni = np.resize(Ni,(3,671))
#    Ni_embedded = tsne.fit_transform(Ni.transpose())
    Ni_embedded = TSNE(n_components=2).fit_transform(Ni.transpose())
    
    title = 'T-SNE visualization of embedding'
    
    movies = get_movies()

    movies = pd.DataFrame(movies[1:], columns = ['movieId', 'title', 'genres'])
    
#    movies = pd.read_csv('movies_cleaned.csv') #loading movies file
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
    
    source = ColumnDataSource(dict(
        x=df_combine['x-tsne'],
        y=df_combine['y-tsne'],
        title= df_combine['title'],
        genre= df_combine['genre']
    ))
    
    plot_lda = figure(plot_width=1000, plot_height=600,
                     title=title, tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                     x_axis_type=None, y_axis_type=None, min_border=1)
    
    plot_lda.scatter(x='x', y='y',source=source,
                    alpha=0.4, size=10)
    
    # hover tools
    hover = plot_lda.select(dict(type=HoverTool))
    hover.tooltips = {"content": "Title: @title, Genre: @genre"}
    
    show(plot_lda)
    save(plot_lda, '{}.html'.format(title))
        
    
      
test1()

