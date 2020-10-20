import numpy as np
import math
import matplotlib.pyplot as plt
import time
from scipy.stats import ks_2samp
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
