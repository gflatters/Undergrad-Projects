import numpy as np
import math
import matplotlib.pyplot as plt
import time
from scipy.stats import ks_2samp

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
