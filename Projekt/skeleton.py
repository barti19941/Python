#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following line in the
entry_points section in setup.cfg:

    console_scripts =
     fibonacci = projekt_1.skeleton:run

Then run `python setup.py install` which will install the command `fibonacci`
inside your current environment.
Besides console scripts, the header (i.e. until _logger...) of this file can
also be used as template for Python modules.

Note: This skeleton file can be safely removed if not needed!
"""
#
# from __future__ import division, print_function, absolute_import
#
# import argparse
# import sys
# import logging
#
# from projekt_1 import __version__
#
# __author__ = "Dawid Bartosiak"
# __copyright__ = "Dawid Bartosiak"
# __license__ = "none"
#
# _logger = logging.getLogger(__name__)
#
#
# def fib(n):
#     """Fibonacci example function
#
#     Args:
#       n (int): integer
#
#     Returns:
#       int: n-th Fibonacci number
#     """
#     assert n > 0
#     a, b = 1, 1
#     for i in range(n-1):
#         a, b = b, a+b
#     return a
#
#
# def parse_args(args):
#     """Parse command line parameters
#
#     Args:
#       args ([str]): command line parameters as list of strings
#
#     Returns:
#       :obj:`argparse.Namespace`: command line parameters namespace
#     """
#     parser = argparse.ArgumentParser(
#         description="Just a Fibonnaci demonstration")
#     parser.add_argument(
#         '--version',
#         action='version',
#         version='projekt_1 {ver}'.format(ver=__version__))
#     parser.add_argument(
#         dest="n",
#         help="n-th Fibonacci number",
#         type=int,
#         metavar="INT")
#     parser.add_argument(
#         '-v',
#         '--verbose',
#         dest="loglevel",
#         help="set loglevel to INFO",
#         action='store_const',
#         const=logging.INFO)
#     parser.add_argument(
#         '-vv',
#         '--very-verbose',
#         dest="loglevel",
#         help="set loglevel to DEBUG",
#         action='store_const',
#         const=logging.DEBUG)
#     return parser.parse_args(args)
#
#
# def setup_logging(loglevel):
#     """Setup basic logging
#
#     Args:
#       loglevel (int): minimum loglevel for emitting messages
#     """
#     logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
#     logging.basicConfig(level=loglevel, stream=sys.stdout,
#                         format=logformat, datefmt="%Y-%m-%d %H:%M:%S")
#
#
# def main(args):
#     """Main entry point allowing external calls
#
#     Args:
#       args ([str]): command line parameter list
#     """
#     args = parse_args(args)
#     setup_logging(args.loglevel)
#     _logger.debug("Starting crazy calculations...")
#     print("The {}-th Fibonacci number is {}".format(args.n, fib(args.n)))
#     _logger.info("Script ends here")
#
#
# def run():
#     """Entry point for console_scripts
#     """
#     main(sys.argv[1:])
#
#
# if __name__ == "__main__":
#     run()

from copy import deepcopy
import numpy as np;
import matplotlib.pyplot as plt;
import random;
from sklearn.cluster import KMeans;

def getColors(k_f):
    Col_f = np.array([(random.uniform(0, 1), random.uniform(0, 1),random.uniform(0, 1)) for i in range(k_f)]);
    return Col_f;

def eucDist(a, b, ax=1):
    return np.linalg.norm(a - b,axis = ax)

def getCentroids(k,x_size_min,x_size_max,y_size_min,y_size_max):
    C = np.array([(random.uniform(x_size_min, x_size_max), random.uniform(y_size_min, y_size_max)) for i in range(k)]);
    return C;

def initBoardUniform(N,min_x,max_x,min_y,max_y):
    X = np.array([(random.uniform(min_x, max_x), random.uniform(min_y, max_y)) for i in range(N)])
    return X

def initBoardGauss(N,k,min_x,max_x,min_y,max_y):
    n = float(N)/k
    X = []
    for i in range(k):
        c = (random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        sx = random.uniform(0.05*max_x,0.25*max_x);
        sy = random.uniform(0.05*max_y,0.25*max_y);
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], sx), np.random.normal(c[1], sy)])
            if a < max_x and a > min_x and b < max_y and b > min_y:
                x.append([a,b])
        X.extend(x)
    X = np.array(X)[:N]
    return X

def minimalizeError (k_f, X_data, C_data, clusters_d,x_min_f, x_max_f, y_min_f, y_max_f):
    tmp_f = 0;
    while (tmp_f != k_f):
        C_data = getCentroids(k_f, x_min_f, x_max_f, y_min_f, y_max_f);
        clusters_d = assignCentroids(X_data, C_data);
        tmp_f = checkClustersMatrix(k_f, clusters_d);
    C_old_data = np.zeros(C_data.shape);
    err = eucDist(C_data, C_old_data, None);
    while err != 0:
        for i in range(len(X_data)):
            distances = eucDist(X_data[i], C_data)
            cluster = np.argmin(distances)
            clusters_d[i] = cluster
        C_old_data = deepcopy(C_data)
        for i in range(k_f):
            points = [X_data[j] for j in range(len(X_data)) if clusters_d[j] == i]
            if(len(points)==0):
                return 0,C_data,clusters_d;
            C_data[i] = np.mean(points, axis=0)
        err = eucDist(C_data, C_old_data, None)
    return 1,C_data,clusters_d;

def plotEndBoard(k_f,X_data,C_data,Col,clusters_d):
    x = X_data[:, 0];
    y = X_data[:, 1];
    c_x = C_data[:,0];
    c_y = C_data[:,1];
    plt.title("Wynik koncowy:");
    plt.xlabel("X");
    plt.ylabel("Y");
    for i in range(k_f):
        plt.scatter(c_x[i], c_y[i], marker='*', s=250, c=Col[i]);
    for i in range(len(X_data)):
        plt.plot(x[i], y[i], 'o', c=Col[int(clusters_d[i])]);
    plt.show();

def plotInitBoard(X_data):
    x = X_data[:, 0];
    y = X_data[:, 1];
    plt.plot(x, y, 'bo');
    plt.title("Punkty w przestrzeni.");
    plt.xlabel("X");
    plt.ylabel("Y");
    plt.show();

def printErrorPlot(k,err):
    plt.title("Zaleznosc bledu srednikowadratowego od ilosci klastrow.");
    plt.xlabel("K");
    plt.ylabel("Blad");
    plt.plot(k,err);
    plt.show();

def assignCentroids(X,C):
    cls = np.zeros(len(X));
    for i in range(len(X)):
        distances = eucDist(X[i], C)
        clust = np.argmin(distances);
        cls[i] = clust;
    return cls;

def checkClustersMatrix(k,clusters):
    tmp = 0;
    for i in range(k):
        for j in range(len(clusters)):
            if clusters[j] == i:
                tmp = tmp+1;
                break;
    return tmp;

def predictK(k_min, k_max, X_data, x_min_f, x_max_f, y_min_f, y_max_f):
    k_range = np.arange(k_min, k_max + 1);
    err_all = np.zeros((k_max - k_min)+1);
    for x in xrange(k_min,k_max+1):
        C_data = np.zeros((2,x));
        clusters_d = np.zeros(len(X_data));
        means = np.zeros(x);
        error = np.zeros(x);
        tmp = 0;
        while (tmp != x):
            C_data = getCentroids(x, x_min_f, x_max_f, y_min_f, y_max_f);
            clusters = assignCentroids(X_data, C_data);
            tmp = checkClustersMatrix(x, clusters);
        C_old_data = np.zeros(C_data.shape);
        err = eucDist(C_data, C_old_data, None);
        while err != 0:
            for i in range(len(X_data)):
                distances = eucDist(X_data[i], C_data)
                cluster = np.argmin(distances)
                clusters_d[i] = cluster
            C_old_data = deepcopy(C_data)
            for i in range(x):
                points = [X_data[j] for j in range(len(X_data)) if clusters_d[j] == i]
                if (len(points)==0):
                    return 0;
                C_data[i] = np.mean(points, axis=0)
            err = eucDist(C_data, C_old_data, None)
        for i in range(x):
            tmp = 0;
            sum = [0,0];
            for j in range(len(clusters)):
                if (clusters[j] == i):
                    sum = sum+X_data[j];
                    tmp = tmp+1;
            means[i] = np.linalg.norm((sum)/tmp);
        for i in range(x):
            for j in range(len(clusters)):
                if (clusters[j] == i):
                    error[i] = error[i] + (np.linalg.norm(X_data[j])-means[i])**2;
        tmp = 0;
        for i in range(len(error)):
            tmp = tmp + error[i];
        err_all[x - 1] = tmp;
    printErrorPlot(k_range,err_all);
    return 1;


x_size_min = -50;
x_size_max = 50;
y_size_min = -50;
y_size_max = 50;

#X_cl = initBoardUniform(1000,x_size_min,x_size_max,y_size_min,y_size_max);
X_cl = initBoardGauss(1000,4,x_size_min,x_size_max,y_size_min,y_size_max);

predict_status = 0;

plotInitBoard(X_cl);

while predict_status != 1:
    predict_status = predictK(1,10,X_cl,x_size_min,x_size_max,y_size_min,y_size_max);


k_cl = int(raw_input("Podaj k: "));
Col_cl = getColors(k_cl);
C_cl = np.zeros((2,k_cl));
clst = np.zeros(len(X_cl))

predict_status = 0;
while predict_status != 1:
    predict_status,C_cl, clst = minimalizeError(k_cl, X_cl, C_cl, clst, x_size_min, x_size_max, y_size_min, y_size_max);

plotEndBoard(k_cl,X_cl,C_cl,Col_cl,clst)

print("Wartosci centroidow z funkcji:");
print(C_cl);
kmeans = KMeans(n_clusters=k_cl)
kmeans = kmeans.fit(X_cl)
labels = kmeans.predict(X_cl)
centroids = kmeans.cluster_centers_
print("Wartosci centroidow z scikitlearn:");
print(centroids);