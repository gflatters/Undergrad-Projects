import numpy as np
import math
import matplotlib.pyplot as plt
import time
from scipy.stats import ks_2samp

'-----------------------------------------------------------------------------'
#font dictionary
title_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'} # Bottom vertical alignment for more space
axis_font = {'fontname':'Arial', 'size':'14'}

'-----------------------------------------------------------------------------'
'''Task 1 functions'''

def sin_transformation(num):
    '''transforms the evenly distributed random numbers to a sin(theta) distribution
        for range 0 < theta < pi. this is the analytic distribution.'''

    x_gen = np.random.uniform(0,1,num)
    x_req = np.arccos(1 - 2*x_gen)
    
    return x_req

def reject_accept(num, ra_num):
    '''uses the reject accept method to produce a sin(theta) distribution
        for range 0 < theta < pi. the number of points tested is 'num', the 
        number of accepted point is 'ra_num'. '''
    first = np.random.uniform(0,math.pi,num) #generates number within required range
    second = np.random.uniform(0,1,num) #value of y
    ra_dist = []
    accept = second < np.sin(first) #the criterion for the number to be accepted

    for i in range(num):
        if accept[i]:
            ra_dist.append(first[i])
            
            #this if statement ensures that the histogram is properly filled
            if len(ra_dist) >= ra_num:
                break
                
    return ra_dist
            
def analytic_histogram(num):
    '''graphs the results of the analytic method as a historgram and performs a KS test on the data'''
    x_req = sin_transformation(num)
    bin_num = 100
    bin_width = math.pi / bin_num
    
    fig = plt.figure(figsize = (10,6))
    ax = fig.add_subplot(111)

    freq, bins, patches = ax.hist(x_req, bin_num, edgecolor="black", linewidth=0.5, label="Sinusoidal random distribution")
    
    #plots the sine curve for comparison
    x = np.linspace(0, math.pi, num=bin_num)
    sine_line = (num/2)*bin_width*np.sin(x)

    plt.plot(x, sine_line, linewidth=5, label="$\\sin(\\theta)$ line")    
    ax.set_xlabel("Random angle, $\\theta$", **axis_font)
    ax.set_xlim(0,math.pi)
    ax.set_ylabel("Frequency", **axis_font)
    ax.set_title("Analytic method $\\sin(\\theta)$ distribution."
                 " Numbers generated = {num}.".format(num=num), **title_font)
    plt.legend()
    plt.show

    print("Analytic method Kolmogorov-Smirnov test results: \n", ks_2samp(freq, sine_line))

    
def ra_histogram(ra_num):
    '''graphs the results of the reject accept method as a historgram and performs a KS test on the data'''
    
    ra_dist = reject_accept(num, ra_num)
    bin_num = 100
    bin_width = math.pi / bin_num
    
    fig = plt.figure(figsize = (10,6))
    ax = fig.add_subplot(111)

    freq, bins, patches = ax.hist(ra_dist, bin_num, edgecolor="black", linewidth=0.5, label="Sinusoidal random distribution")
    
    #plots the sine curve for comparison, using ra_num to ensure it fits with the 
    #amount of accepted numbers
    x = np.linspace(0, math.pi, num=bin_num)
    sine_line = (ra_num/2)*bin_width*np.sin(x)
    
    plt.plot(x, sine_line, linewidth=5, label="$\\sin(\\theta)$ line")   
    ax.set_xlabel("Random angle, $\\theta$", **axis_font)
    ax.set_xlim(0,math.pi)
    ax.set_ylabel("Frequency", **axis_font)
    ax.set_title("Reject accept method $\\sin(\\theta)$ distribution."
                 " Numbers generated = {ra_num}.".format(ra_num=ra_num), **title_font)
    plt.legend()
    plt.show
    
    print("Reject-accept method Kolmogorov-Smirnov test results: \n", ks_2samp(freq, sine_line))
    
def method_timer():
    '''plots a graph comparing the speed of the two tests for varying amounts 
        of generated numbers, 25 repeats to reduce effect of fluctuations.'''
    num_list = []
    
    analytic_time = []
    analytic_total = []
    
    ra_time = []
    ra_total = []
     
    num_iter = 100
    num_max = 10000000
    while num_iter<num_max:
        
        i = 1
        repeat = 25
        #timing done for 25 repeats to reduce fluctuations in results
        while i<repeat:
            
            start = time.time()
            a = sin_transformation(num_iter) #waste variable to run once
            end = time.time()
            analytic_time.append(end-start)
            
            start = time.time()
            b = reject_accept(num_iter, ra_num) #waste variable to run once
            end = time.time()
            ra_time.append(end-start)
            
            i += 1

        analytic_total.append(np.sum(analytic_time))#sums all the times together
        ra_total.append(np.sum(ra_time))#sums all the times together
        num_list.append(num_iter)
            
        num_iter *= 10
        
    fig = plt.figure(figsize = (10,6))
    ax = fig.add_subplot(111)
    
    ax.set_xlabel("Random numbers generated", **axis_font)
    ax.set_xscale('log')
    ax.set_ylabel("Time (s)", **axis_font)
    ax.set_title("Method time comparison", **title_font)
    plt.plot(num_list, analytic_total, label = "Analytic method", marker = "d")
    plt.plot(num_list, ra_total, label = "Reject-accept method", marker = "o")
    plt.legend()
    plt.show

'-----------------------------------------------------------------------------'
'''Task 1 show in console'''

print("Printing results of task 1... \n")
num = 10000000 #memory error for values above 10^8
ra_num = 1000000 #'num' should be 10 times larger so there are a sufficient amount of accepts

analytic_histogram(num)
ra_histogram(ra_num)

method_timer()
