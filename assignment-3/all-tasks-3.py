# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:32:34 2019

@author: George Flatters
"""

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

'-----------------------------------------------------------------------------'
'''functions for physics problem 1'''

def decay_simulation(nuclei_num):
    '''produces an exponential distribution of the lifetimes and positions of 
        the nuclei decay events'''
    
    lifetime = 550e-6
    speed = 2000 #nuclei travelling at 2000 m/s
    
    time_dist = np.random.exponential(scale=lifetime, size=nuclei_num)
    position_dist = time_dist*speed
            
    return time_dist, position_dist

def decay_num():
    '''finds the amount of nuclei that decay before reaching the detector. this 
        number will be used when plotting the rest of the data for this problem.'''
    time_dist, position_dist = decay_simulation(nuclei_num)
    
    #nuclei must decay at a position smaller than 2m to be detected
    decayed_nuclei = len([i for i in position_dist if i <= 2])
    
    return decayed_nuclei

def graph_distributions(nuclei_num):
    '''plots the exponential lifetime and position distribution with a fitted model curve.
        states how many nuclei decay before they reach the detector'''
    
    time_dist, position_dist = decay_simulation(nuclei_num)
    
    #variables
    bin_num = 200 #greater bin number means freq of bins tends to theory line
    lifetime = 550e-6
    speed = 2000
    max_time = 0.004
    max_position = max_time*speed
    
    #independent variables for plotting theoretical lines
    t_vals = np.linspace(0, max_time, num=bin_num)
    x_vals = speed*t_vals
    
    fig = plt.figure(figsize = (10,6))
    ax = fig.add_subplot(111)
    
    nuclei_freq, bins, patches = ax.hist(time_dist, bin_num, edgecolor="black", color="green", linewidth=0.5, label="Random exponential distribution")
    
    nuclei_left = nuclei_freq[0]*np.exp(-t_vals/lifetime)
    ax.plot(t_vals, nuclei_left, linewidth=5, color="orange", label="Theoretical curve")
    
    ax.set_xlabel("Lifetime (s)", **axis_font)
    ax.set_xlim(0,max_time)
    ax.set_ylabel("Number of nuclei", **axis_font)
    ax.set_title("Nuclei lifetime distribution", **title_font)
    plt.legend()
    plt.show
    
    fig = plt.figure(figsize = (10,6))
    ax = fig.add_subplot(111)
    
    position_freq, bins, patches = ax.hist(position_dist, bin_num, edgecolor="black", color="blue", linewidth=0.5, label="Random exponential distribution")
    
    position_model = position_freq[0]*np.exp(-t_vals/lifetime)
    ax.plot(x_vals, position_model, linewidth=5, color="orange", label="Theoretical curve")
    
    ax.set_xlabel("Decay position (m)", **axis_font)
    ax.set_xlim(0,max_position)
    ax.set_ylabel("Number of nuclei", **axis_font)
    ax.set_title("Nuclei decay position distribution", **title_font)
    plt.legend()
    plt.show
    
    decayed_nuclei = decay_num()
    
    percent_detected = decayed_nuclei/len(position_dist)
    
    print("The percentage of nuclei that decay before reaching the detector is {percent_detected}. "
          "Total nuclei detected = {decayed_nuclei}."
          .format(percent_detected=percent_detected, decayed_nuclei=decayed_nuclei))
    

def get_angles(decayed_nuclei):
    '''the distribution of angles as discussed in the report'''
    
    decayed_nuclei = decay_num()
    #theta is evenly distributed between 0 and 2*pi
    theta = 2*np.pi*np.random.random(decayed_nuclei)
    
    #phi must be sinusoidally distributed between 0 and pi
    #using analytic method from task 1
    phi = np.arccos(2*np.random.random(decayed_nuclei) - 1)

    phi -= np.pi/2 #theta = 0 must be pointing along the x-axis towards detector
    
    return phi, theta

def get_detector(decayed_nuclei):
    '''defines the 2d position of the detector using trigonometry,
        converting spherical coordiantes to cartesian as discussed in report.'''
    
    phi, theta = get_angles(decayed_nuclei)
    d = 2 #detector is 2m from injection point

    x_vals = d*np.tan(phi)*np.cos(theta)
    y_vals = d*np.tan(phi)*np.sin(theta)

    return x_vals, y_vals

def get_xsmear(x_dist, res_x):
    '''simulates resolution by smearing each value using a gaussian distribution.
        the value is the mean of the distribution'''

    x_dist += np.random.normal(loc=x_dist, scale=res_x)

    return x_dist

def get_ysmear(y_dist, res_y):
    '''simulates resolution by smearing each value using a gaussian distribution.
        the value is the mean of the distribution'''

    y_dist += np.random.normal(loc=y_dist, scale=res_y)

    return y_dist

def apply_smear(decayed_nuclei, res_x, res_y):
    '''applies the gaussian smearing to the bins'''
    
    x_vals, y_vals = get_detector(decayed_nuclei)

    x_smear = np.copy(x_vals)
    y_smear = np.copy(y_vals)
    
    x_smear = np.apply_along_axis(get_xsmear, 0, x_smear, res_x)
    y_smear = np.apply_along_axis(get_ysmear, 0, y_smear, res_y)
    
    return x_smear, y_smear
    
def graph_angles(decayed_nuclei):
    '''graphs the angle distributions just to test that they're distributed correctly'''
    phi, theta = get_angles(decayed_nuclei)
    bin_num = 200
    
    fig = plt.figure(figsize = (10,6))
    ax = fig.add_subplot(111)

    h = ax.hist2d(phi, theta, bins=bin_num)
    ax.set_xlabel("theta", **axis_font)
    ax.set_ylabel("phi", **axis_font)
    ax.set_title("Distribution of decay angles", **title_font)
    
    plt.colorbar(h[3], ax=ax, label = "Frequency")
    plt.show()

def graph_detector(decayed_nuclei, res_x, res_y):
    '''graphs the detector readings with and without smear'''
    
    detector_range = np.array([[-2, 2], [-2, 2]]) #defines the dimensions of the detector
    
    x_vals, y_vals = get_detector(decayed_nuclei) #plot without smear
    x_smear, y_smear = apply_smear(decayed_nuclei, res_x, res_y) #plot with smear
    
    #bin_num = 200, can use this to show the "real distribution" without uncertainty from resolution
    
    #bin number used reflect resolution of the detector
    bin_num = (((detector_range[0][1] - detector_range[0][0])/res_x),
                ((detector_range[1][1] - detector_range[1][0]))/res_y)
    
    fig = plt.figure(figsize = (10,6))
    ax = fig.add_subplot(111)

    h = ax.hist2d(x_vals, y_vals, bins=bin_num, range=detector_range, cmap=cm.hot)
    ax.set_xlabel("x (m)", **axis_font)
    ax.set_ylabel("y (m)", **axis_font)
    ax.set_title("Detector reading without smear", **title_font)
    
    plt.colorbar(h[3], ax=ax, label = "Frequency")
    plt.show()
    
    fig = plt.figure(figsize = (10,6))
    ax = fig.add_subplot(111)

    h = ax.hist2d(x_smear, y_smear, bins=bin_num, range=detector_range, cmap=cm.hot)
    ax.set_xlabel("x (m)", **axis_font)
    ax.set_ylabel("y (m)", **axis_font)
    ax.set_title("Detector reading with smear", **title_font)
    
    plt.colorbar(h[3], ax=ax, label = "Frequency")
    plt.show()
    
    #notice difference in the frequency values of the color bars in the two graphs
'-----------------------------------------------------------------------------'
'''physics problem 1 show in console'''

print("Printing the results of the first physics problem... \n")
nuclei_num = 10000000 #total number of nuclei
decayed_nuclei = decay_num()

#resolution of detector array
res_x = 0.1
res_y = 0.3

graph_distributions(nuclei_num)
graph_angles(decayed_nuclei)
graph_detector(decayed_nuclei, res_x, res_y)

'-----------------------------------------------------------------------------'
'''functions for physics problem 2'''

def graph_events(total_events, x_rounded):
    '''will be used to plot the poisson distribution of total events for the cross 
        section at which the confidence level is 0.95.'''    
    bin_num = 25
    
    fig = plt.figure(figsize = (10,6))
    ax = fig.add_subplot(111)

    ax.hist(total_events, bin_num, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Events")
    ax.set_ylabel("Frequency")
    ax.set_title("Event distribution for a cross section of {x_rounded}nb".format(x_rounded=x_rounded))
    plt.show
    
def get_xsection(L_uncertainty = False):
    '''performs N pseudo experiments to find the cross section at which the confidence
        level is 95% within a given error. gives the option of introducing integrated luminosity uncertainty.'''
    
    N = 1000000 #number of pseudo experiments, heavily effects computational cost
    accuracy = 1e-4 #used to increment x_section, defines level of accuracy
    error = accuracy/2 #error is half the smallest increment
    
    print("Finding the cross section at which the confidence "
          "level is 95%, to an accuracy of +/- {error}...".format(error=error))
    
    #initial guesses, optimised depending on L_uncertainty true/false
    if L_uncertainty:
        x_section = 0.435
        
    else:
        x_section = 0.399
    
    while x_section < 0.6:
        
        #gaussian uncertainty on the background prediction
        background = np.random.normal(5.7, 0.4, size = N) 
        
        #poisson variation in the background
        background_count = np.random.poisson(background, size = N)
        
        #example of how uncertainty in the luminosity value can be added, uses L = 12 +/- 0.3
        if L_uncertainty:
            
            L_mean = np.random.normal(12, 0.3, size = N)
            L = np.random.poisson(L_mean, size = N)
            
        else:  
            L = 12 #nanobarns (10^-28 m)
            
        expected = L*x_section
        
        #poisson variation in the signal production
        signal_count = np.random.poisson(expected, size = N)
        
        #total events detected
        total_events = signal_count + background_count
        
        #extracts every value in total_events that is greater than 5
        above_5 = [i for i in total_events if i > 5]
        
        #ratio of values above 5 to total values
        confidence = len(above_5) / len(total_events)
        
        #rounds up the cross section value in accordance with the accuracy
        x_rounded = round(x_section, 5)

        
#        print("Confidence level is {confidence} for a cross section "
#              "of {x_rounded}".format(confidence=confidence, x_rounded=x_rounded))
        
        #break if confidence is found to be within 1e-5 of 0.95
        if np.isclose(confidence, 0.95, accuracy):
            print("The required cross section for a confidence level of 0.95 is {x_rounded} +/- {error} " 
            "nanobarns.".format(x_rounded=x_rounded, error=error))
            graph_events(total_events, x_rounded)
            break
        
        x_section += accuracy
        
def graph_confidence():
    '''graphs confidence against cross section for a large range of cross sections. 
        lines are plotted for confidence = 0.95 and for the cross section found 
        in the previous function to show overlap with curve.'''
        
    print("Printing confidence against cross section graph...")
    
    confidence_list = []
    x_section = []
    
    N = 100000 #smaller N than previously so less costly
    x_vals = 0.00 #initial guess for cross section
    x_step = 0.01
    x_max = 0.81
    
    while x_vals < x_max:
        
        #gaussian uncertainty on the background prediction
        background = np.random.normal(5.7, 0.4, size = N) 
        
        #poisson variation in the background
        background_count = np.random.poisson(background, size = N)
        
        L = 12
        expected = L*x_vals
        
        #poisson variation in the signal production
        signal_count = np.random.poisson(expected, size = N)
        
        #total events detected
        total_events = signal_count + background_count
        
        #extracts every value in total_events that is greater than 5
        above_5 = [i for i in total_events if i > 5]
        
        #ratio of values above 5 to total values
        confidence = len(above_5) / len(total_events)
        
        confidence_list.append(confidence)
        x_section.append(x_vals)
        
        x_vals += x_step
     
    fig = plt.figure(figsize = (10,6))
    ax = fig.add_subplot(111)
    
    #pseudo data for a dotted line to represent confidence = 0.95
    dot_x = np.linspace(0,1)
    dot_y = np.array([0.95 for i in dot_x])
    
    #intersection line
    intersect_y = np.linspace(0.5, 1)
    intersect_x = np.array([0.404 for i in intersect_y])
    
    ax.plot(x_section, confidence_list)
    ax.plot(dot_x, dot_y, 'r--')
    ax.plot(intersect_x, intersect_y, 'r--')
    ax.set_xlabel("Cross section (nb)")
    ax.set_xlim(0, (x_max-0.01))
    ax.set_ylabel("Confidence")
    ax.set_ylim(0.5, 1)
    ax.set_title("Confidence vs cross section for {N} pseudo-experiments".format(N=N))
    plt.show 

'-----------------------------------------------------------------------------'
'''physics problem 2 show in console'''

print("Printing the results of physics problem 2...")
get_xsection(L_uncertainty = False)
#get_xsection(L_uncertainty = True)
graph_confidence()