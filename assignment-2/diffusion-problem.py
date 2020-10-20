import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from matplotlib import cm 
from scipy import optimize
from scipy.stats import linregress

'-----------------------------------------------------------------------------'
'''Functions for solving problem 2'''

def impose_furnace(poker, xnum):
    '''furnace boundary condition, T = 1273.15'''
    poker[:,0] = 1273.15

    return poker

def impose_ice(poker, xnum):
    '''ice boundary condition, T = 273.15'''
    poker[:,xnum-1] = 273.15

    return poker

def test_equilibrium(poker, evolved_poker, xnum, width, tolerance, num_env_nodes, iter_num):
    '''checks if all nodes have reached equilibrium'''

    equilib_check = np.abs(evolved_poker - poker) < tolerance

    num_nodes = xnum*width
    num_equi = np.sum(equilib_check)

    sys.stdout.flush()
    sys.stdout.write("\rNumber of nodes at equilibrium: %5d / %5d;"\
                     "\tNumber of iterations completed: %6d" % 
                     (num_equi, num_nodes - num_env_nodes, 
                      iter_num+1))

    if num_equi >= num_nodes - num_env_nodes:
        print
        return True

    return False

def iterate(poker, xnum, prefactor_inv):
    '''solves the linear system of equations derived from equation 3, 
        advances the time by the time step. uses the inverse as it has
        already been calculated.'''
    poker_new = poker.dot(prefactor_inv)    
    return poker_new

def evolve(poker, xnum, prefactor_inv, num_iters, cold=False):
    '''evolves the system a set number of iterations, used to show time evolution'''

    impose_furnace(poker, xnum) #hot end imposed for both cases

    if cold:
        impose_ice(poker, xnum) #for case 2
    
    for counter in range(num_iters):

        evolved_poker = iterate(poker, xnum, prefactor_inv)

        poker = np.copy(evolved_poker)

        impose_furnace(poker, xnum)

        if cold:
            impose_ice(poker, xnum)
    
    print
    return poker 

def get_equilibrium(poker, xnum, width, prefactor_inv, tolerance, num_env_nodes, cold=False, max_iters=1e6):
    '''evoles the system until equilibrium is reached to a specified tolerance'''
    max_iters = int(max_iters)

    impose_furnace(poker, xnum)

    if cold:
        impose_ice(poker, xnum)

    for counter in range(max_iters):

        evolved_poker = iterate(poker, xnum, prefactor_inv)

        if test_equilibrium(poker, evolved_poker, xnum, width, tolerance, num_env_nodes, counter):
            print( "\nReach equilibrium after {} iterations.".format(counter+1))
            return evolved_poker, counter

        poker = np.copy(evolved_poker)

        impose_furnace(poker, xnum)

        if cold:
            impose_ice(poker, xnum)
        # Boundary conditions need to be reimposed after each iteration.

    print ("\nEquilibrium not found after {} iterations.".format(counter))
    return False, False

'-----------------------------------------------------------------------------'
'''constants and linear algebra routine'''
# Constants
initial_temp = 293.15
furnace_temp = 1273.15
ice_temp = 273.15
diffusivity = 23e-6

# Chosen grid variables
xnum = 50 # poker is 50cm long 
spacing = 0.01 

#width of poker in cm, this doesn't change diffusion data but makes it 
#more representative of a physical situation
width = 5

error = 1e-4 #absolute error of temperature (K)

timestep = 1 #changes the number of iterations required to reach equilibrium
#increase this for shorter run time

poker = np.zeros((width, xnum))
poker.fill(initial_temp) # poker starts off at a temperature of 293K

prefactor = np.zeros((xnum, xnum))
for i in range(xnum):
    for j in range(xnum):
        if i == j:
            prefactor[i,j] = 1 + 2*diffusivity*timestep / spacing**2
        if i == j-1 or i == j+1:
            prefactor[i,j] = - diffusivity*timestep / spacing**2

#the bottom and top boundary nodes only average over one neighbour            
prefactor[0,0] = 1 + diffusivity*timestep / spacing**2
prefactor[xnum-1,xnum-1] = 1 + diffusivity*timestep / spacing**2

prefactor_inv = np.linalg.inv(prefactor)
'-----------------------------------------------------------------------------'
'''functions that produce results for the two cases'''

def case_1():
    '''prints results of case 1'''
    print("Case 1: ")
    num_env_nodes = width # Hot end only
    equi_solution, iters_needed = get_equilibrium(poker, xnum, width,
                                                         prefactor_inv, error, 
                                      num_env_nodes, cold=False)    
    
    x = np.linspace(0, xnum, xnum)
    fig = plt.figure(figsize = (14,8))
    ax = fig.add_subplot(111)
#    ax.plot(x, equi_solution[0,:], label='{iters_needed} iterations'.format(iters_needed = iters_needed))
#    #plots the equilibrium solution
    iter_inc = 1000
    while iter_inc<iters_needed:
        #shows the time evolution
        num_iters = int(math.ceil(iter_inc / timestep))
        solution = evolve(poker, xnum, prefactor_inv, num_iters, cold=False)
        
        ax.plot(x, solution[0,:], label='{iter_inc} iterations'.format(iter_inc = iter_inc))
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("Temperature (K)")
        ax.set_xlim(0, 50)
        ax.set_ylim(initial_temp)
        plt.yticks(np.arange(initial_temp, furnace_temp, 150))
        ax.set_title("Case 1: temperature distribution of poker")
        plt.legend()
        iter_inc += 2000
        
    plt.show()
    
def case_2():
    '''prints results of case 2'''
    print("Case 2: ")
    num_env_nodes = width*2 # both hot and cold end
    equi_solution, iters_needed = get_equilibrium(poker, xnum, width,
                                                         prefactor_inv, error, 
                                      num_env_nodes, cold=True)    
    
    x = np.linspace(0, xnum, xnum)
    fig = plt.figure(figsize = (14,8))
    ax = fig.add_subplot(111)
#    ax.plot(x, equi_solution[0,:], label='{iters_needed} iterations'.format(iters_needed = iters_needed))
#    #plots the equilibrium solution
    iter_inc = 100
    while iter_inc<iters_needed:
        #shows the time evolution
        num_iters = int(math.ceil(iter_inc / timestep))
        solution = evolve(poker, xnum, prefactor_inv, num_iters, cold=True)
        
        ax.plot(x, solution[0,:], label='{iter_inc} iterations'.format(iter_inc = iter_inc))
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("Temperature (K)")
        ax.set_xlim(0, 50)
        ax.set_ylim(ice_temp)
        plt.yticks(np.arange(ice_temp, furnace_temp, 200))
        ax.set_title("Case 2: Temperature distribution of poker")
        plt.legend()
        iter_inc += 500
        
    plt.show()
    
'-----------------------------------------------------------------------------'
'''problem 2 print results to console'''

print("Calculating the results of the second physics problem...\n")  
case_1()
case_2()
