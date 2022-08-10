import numpy as np
import matplotlib.pyplot as plt
def increasing(x,a,b):
    if(x<=a):
        r=0
    elif(x>=b):
        r=1
    else:
        r=(x-a)/(b-a)
    return r
def gauss(x,u,s):
    r = np.exp(((-(x - u) ** 2) / (2 * s ** 2)))
    return(r)

def gauss1(x,u,s):
    if (x <= u):
        r = 1
    else:
        r = np.exp(((-(x - u) ** 2) / (2 * s ** 2)))
    return(r)

def gauss2(x,u,s):
    if (x >= u):
        r = 1
    else:
        r = np.exp(((-(x - u) ** 2) / (2 * s ** 2)))
    return(r)

def decreasing(x,a,b):
    if (x <= a):
        r = 1
    elif (x >= b):
        r = 0
    else:
        r = (b-x) / (b - a)
    return r

def trape(x,a,b,c,d):
    if (x <= a or x>=d):
        r = 0
    elif (x >= b and x<=c):
        r = 1
    elif (x> a and x<b):
        r = (x-a) /(b - a)
    else:
        r = (d-x)/(d-c)
    return r

