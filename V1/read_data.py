# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:48:05 2018

@author: selen
"""
import numpy as np 
import codecs

def build_movies_dict(movies_file):
    i = 0
    movie_id_dict = {}
    with codecs.open(movies_file, 'r', 'latin-1') as f:
        for line in f:
            if i == 0:
                i = i+1
            else:
                print(line)
                movieId,title,genres = line.split(',')
                movie_id_dict[int(movieId)] = i-1
                i = i +1
    return movie_id_dict

def read_data(input_file,movies_dict):
    #no of users
    users = 671
    #users = 5
    #no of movies
    movies = 9125
    #movies = 135887
    X = np.zeros(shape=(users,movies))
    i = 0
    #X = genfromtxt(input_file, delimiter=",",dtype=str)
    with open(input_file,'r') as f:
        for line in f:
            if i == 0:
                i = i +1
            else:
                #print "i is",i
                user,movie_id,rating,timestamp = line.split(',')
                #get the movie id for the numpy array consrtruction
                print(user,movie_id,rating,timestamp)
                print(movies_dict)
                id = movies_dict[int(movie_id)]
                
                #print "user movie rating",user, movie, rating, i
                X[int(user)-1,id] = float(rating)
                i = i+1
    return X