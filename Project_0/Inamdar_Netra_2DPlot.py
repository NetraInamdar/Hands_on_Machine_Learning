# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 16:32:00 2019

@author: Netra
"""
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

rows=150
#filename=input("Enter name of input file:")
filename='IrisData.txt'
fin=open(filename,"r")
data = np.zeros([rows, 5])

for k in range(rows):
    aString = fin.readline()
    t = aString.split("\t")
    for j in range(5):
        if (str(t[j])=='setosa\n'):
            t[j]=1
        elif (str(t[j])=='versicolor\n'):
            t[j]=2
        elif (str(t[j])=='virginica\n'):
            t[j]=3
        data[k,j] = float(t[j])
fin.close()

for k in range(rows):
    if data[k,4] == 1:
        plt.scatter(data[k,0], data[k,2], color = "red",
        marker = "o", label = "Setosa")
    elif data[k,4] == 2:
        plt.scatter(data[k,0], data[k,2], color = "green",
        marker = "v", label = "Versicolor")
    elif data[k,4] == 3:
        plt.scatter(data[k,0], data[k,2], color = "blue",
        marker = "v", label = "Verginica")
        
    plt.xlabel("This is Sepal Length (x axis)")
    plt.ylabel("This is Petal Length (y axis)")
    plt.title("Sepal Length VS Petal Length")
    
plt.savefig('Inamdar_Netra_MyPlot.png')


