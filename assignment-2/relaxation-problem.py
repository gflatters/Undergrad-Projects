import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from matplotlib import cm 
from scipy import optimize
from scipy.stats import linregress

'-----------------------------------------------------------------------------'
'''functions for solving physics problem 1'''
    
def iterate_node(grid, i, j, num_x, num_y):
    '''function that applies the iteration technique (equation 2)
        for each element of a grid. value is saved into new_grid.'''
    top = grid[i,j+1] if j < num_y-1 else 0
    down = grid[i,j-1] if j > 0 else 0
    right = grid[i+1,j] if i < num_x-1 else 0
    left  = grid[i-1,j] if i > 0 else 0

    return (top + down + left + right)*0.25

def get_jacobi(grid, num_x, num_y):
    '''applies my node iteration function to grid. the new grid values are saved
        into a seperate grid.'''

    grid_new = np.zeros((num_x, num_y))
    
    for i in range(0,num_x):
        for j in range(0,num_y):
            
            #values being saved into a seperate grid
            grid_new[i,j] = iterate_node(grid, i, j, num_x, num_y)
            
    return grid_new

def get_gs(grid, num_x, num_y):
    '''applies my node iteration fuction to grid. the new grid values are saved
        into the origianl grid. this method converges twice as fast as jacobi.'''
    grid_new = np.copy(grid)#values being saved into the original grid
    
    for i in range(0,num_x):
        for j in range(0,num_y):
            
            #iterate_node takes grid_new as an argument, saving values into original grid
            grid_new[i,j] = iterate_node(grid_new, i, j, num_x, num_y)
        
    return grid_new

def get_capacitor(grid, num_x, num_y):
    '''function that creates two capacitor plates. for the report d will be kept constant
        and a will be varied to show the field configuration approaching the infinite
        plate solution.'''    
        
    #capacitor with larger a/d shows approach to infinite plate solution
    a = 10 #width of the capacitor
    d = 10 #distance between plates
    grid[num_x//2-d//2,num_y//2 - a//2:num_y//2 + a//2] = 1
    grid[num_x//2+d//2,num_y//2 - a//2:num_y//2 + a//2] = -1
    
    return 2*a
    
def test_convergence(diff, given_error, num_x, num_y, num_boundary_nodes, iter_num):
    '''tests for convergence for a given error'''

    num_nodes = num_x*num_y

    conv = diff < given_error

    #Number of converged entries is sum of all true values of convergence matrix
    num_converged = np.sum(conv)
   
    #Set boundary nodes are reset so that they don't converge
    if num_converged >= num_nodes - num_boundary_nodes:
        print
        return True

    return False

def jacobi_solution(num_x, num_y, err_total=1e-5, timer=False):
    '''carries out the jacobi method until a convergence condition is satisfied to
        solve the laplace equation. if timer=True then this function will be used
        to count the number of iterations required to reach the convergence condition. 
        otherwise it will be uesd to plot a graph (return grid).'''
    grid = np.random.rand(num_x,num_y)
    grid_new = np.zeros((num_x,num_y))
    diff = np.zeros((num_x,num_y))
    
    boundary_nodes = get_capacitor(grid, num_x, num_y)
    
    #iteration procedure. number of iterations required for convergence condition = iter_num
    #cannot exceed iter_max
    iter_num = 1
    iter_max=1e6
    while iter_num <= iter_max:
        
        grid_new = get_jacobi(grid, num_x, num_y)
        
        diff = np.abs(grid_new - grid)
        
        iter_num += 1
        
        if test_convergence(diff, err_total, num_x, num_y, 
                             boundary_nodes, iter_num):
            print("Jacobi method: convergence condition satisfied for all nodes "\
                      "after {iters} iterations. Error = {tolerance}, J = {J}. ".format(
                        tolerance=err_total,iters=iter_num, J=num_x))
                
            if timer == True:
                return iter_num
            else :
                return grid
        
        grid = np.copy(grid_new)
        #boundary conditions constant for all iterations
        get_capacitor(grid, num_x, num_y)
    
    print("Convergence condition not satisfied after {iter_num} iterations. "\
      "Error = {tolerance}, J = {J}.".format(tolerance=err_total,iter_num=iter_num, J=num_x))
                        
    return False

def gs_solution(num_x, num_y, err_total=1e-5, timer=False):
    '''same as previous function but using the gauss-seidel method'''
    grid = np.random.rand(num_x,num_y)
    grid_new = np.zeros((num_x,num_y))
    diff = np.zeros((num_x,num_y))
    
    boundary_nodes = get_capacitor(grid, num_x, num_y)
    
    #iteration procedure. number of iterations required for convergence condition = iter_num
    iter_num = 1
    iter_max=1e6
    while iter_num <= iter_max:
        
        grid_new = get_gs(grid, num_x, num_y)
        
        diff = np.abs(grid_new - grid)
        
        iter_num += 1
        
        if test_convergence(diff, err_total, num_x, num_y, 
                             boundary_nodes, iter_num):
            print("Gauss seidel method: convergence condition satisfied for all nodes "\
                      "after {iters} iterations. Error = {tolerance}, J = {J}.\n ".format(
                        tolerance=err_total,iters=iter_num, J=num_x))
            
            if timer == True:
                return iter_num
            else :
                return grid
    
        grid = np.copy(grid_new)
        #boundary conditions constant for all iterations
        get_capacitor(grid, num_x, num_y)
    
    print("Convergence condition not satisfied after {iter_num} iterations. "\
      "Error = {tolerance}, J = {J}.".format(tolerance=err_total,iter_num=iter_num, J=num_x))
    return False
'-----------------------------------------------------------------------------'
'''functions for plotting the results of the first physics problem
    using the gauss siedel method'''

def plot_potential_field_gs(num_x, num_y, step_size):
    '''function that ppokeruces a contour plot of the potential field.
        plots the vector gradient of the potential field imposed on top.'''

    x = np.linspace(0, num_x*step_size, num_x)
    y = np.linspace(0, num_y*step_size, num_y)  
    
    X1, Y1 = np.meshgrid(x,y)
    
    Y2, X2 = np.mgrid[0:num_x*step_size:complex(0,num_x),
                    0:num_y*step_size:complex(0,num_y)]
    
    fig, ax = plt.subplots(figsize = (18,10))
    
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Potential field contour plot")
    
    plt.contourf(X1, Y1, np.transpose(gs_solution(num_x, num_y)), 100, rstride=1, cstride=1,
                             cmap=cm.viridis, linea=0)
    
    cbar = plt.colorbar(label = "Potential (V)")
    cbar.solids.set_edgecolor("face")
    
    plt.contour(X2, Y2, np.transpose(gs_solution(num_x, num_y)), cmap=cm.Wistia)
    plt.show()
   
def plot_electric_field_gs(num_x, num_y, step_size):
    '''function that plots the magnitude of the electric field as a contour map.
        plots the vector gradient of the potential field imposed on top.'''
    x = np.linspace(0, num_x*step_size, num_x)
    y = np.linspace(0, num_y*step_size, num_y)  
    X1, Y1 = np.meshgrid(x,y)
    
    Y2, X2 = np.mgrid[0:num_x*step_size:complex(0,num_x),
                    0:num_y*step_size:complex(0,num_y)]
    
    #vector gradient values
    vector_u, vector_v = np.gradient(np.transpose(gs_solution(num_x, num_y)))

    vector_u = np.zeros((num_x, num_y)) - vector_u
    vector_v = np.zeros((num_x, num_y)) - vector_v

    magnitude = np.sqrt(vector_u*vector_u + vector_v*vector_v)

    lw = 5*magnitude/magnitude.max()
    
    fig, ax = plt.subplots(figsize = (18,10))
    
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Electric field contour plot")

    plt.contourf(X1, Y1, magnitude, 100, rstride=1, cstride=1, cmap=cm.viridis, linea=0)
    
    cbar = plt.colorbar(label = "E (Vm^-1)")
    cbar.solids.set_edgecolor("face")                            

    plt.streamplot(X2, Y2, vector_v, vector_u, color=magnitude, 
                                 linewidth=lw, cmap=cm.Wistia)
    plt.show()

#def plot_vector(num_x, num_y, step_size):
#    '''only plots the vector gradient lines'''
#
#    Y, X = np.mgrid[0:num_x*step_size:complex(0,num_x),
#                    0:num_y*step_size:complex(0,num_y)]
#
#    vector_u, vector_v = np.gradient(np.transpose(gs_solution(num_x, num_y)))
#
#    vector_u = np.zeros((num_x, num_y)) - vector_u
#    vector_v = np.zeros((num_x, num_y)) - vector_v
#
#    magnitude = np.sqrt(vector_u*vector_u + vector_v*vector_v)
#
#    plt.figure(figsize = (18,10))
#
#    lw = 5*magnitude/magnitude.max()
#
#    plt.streamplot(X, Y, vector_v, vector_u, color=magnitude, 
#                                 linewidth=lw, cmap=cm.hot)
#    plt.xlabel("x")
#    plt.ylabel("y")
#    plt.title("Vector gradient of potential field")
#    cbar = plt.colorbar(label = "Potential (V)")
#    cbar.solids.set_edgecolor("face")
#    plt.draw()

'-----------------------------------------------------------------------------'
'''functions of the models discussed in the report to fit the iteration vs J data to'''

def jacobi_Jmodel(j, p): #j is the independent variable
    return 0.5*p*j**2

def gs_Jmodel(j, p): #j is the independent variable
    return 0.25*p*j**2

'-----------------------------------------------------------------------------'
'''functions for measuring the iterations required for varying convergence conditions
    and values of J, where the grid has dimensions JxJ.'''

def jacobi_convergence_timer(err_list1, jacobi_iter_list1):
    '''measures iterations required for different convergence conditions using the
        jacobi method. line of best fit plotted (x axis is log scale).
        gradient and standard error used in analysis.'''
    x_num = 30 #value of J constant for this data
    y_num = 30
    i = 1e-8
    err_max = 1e-1
    while i<=err_max:
        err_val = i
        err_list1.append(err_val)
        
        jacobi_iter_num = jacobi_solution(x_num, y_num, err_total=err_val, timer=True)
        jacobi_iter_list1.append(jacobi_iter_num)
        
        i *= 10
   
    fig = plt.figure(figsize = (14,8))
    ax = fig.add_subplot(111)
    
    err_points=np.abs(np.log10(err_list1))#this outputs the value of p where err_val = 1e-p
    slope, intercept, r_value, p_value, std_err = linregress(err_points, jacobi_iter_list1)#line of best fit
    
    ax.set_xlabel("p (error = 1x10^-p)")
    ax.set_ylabel("Iterations")
    ax.set_title("Jacobi method: Convergence test")
    plt.scatter(err_points, jacobi_iter_list1, label = "Jacobi method", marker = "d")
    plt.plot(err_points, intercept + slope*err_points, 'r', label='Line of best fit')
    plt.plot([], [], label="Gradient = {slope}".format(slope=slope))
    plt.plot([], [], label="Standard error = {std_err}".format(std_err=std_err))
    plt.legend()
    plt.show
    
def gs_convergence_timer(err_list2, gs_iter_list1):
    '''measures iterations required for different convergence conditions using the
        gauss-seidel method. line of best fit plotted (x axis is log scale).
        gradient and standard error used in analysis.'''
    x_num = 30 #value of J constant for this data
    y_num = 30
    i = 1e-8
    err_max = 1e-1
    while i<=err_max:
        err_val = i
        err_list2.append(err_val)
        
        gs_iter_num = gs_solution(x_num, y_num, err_total=err_val, timer=True)
        gs_iter_list1.append(gs_iter_num)
        
        i *= 10
    
    fig = plt.figure(figsize = (14,8))
    ax = fig.add_subplot(111)
    
    err_points=np.abs(np.log10(err_list2)) #again outputs p
    slope, intercept, r_value, p_value, std_err = linregress(err_points, gs_iter_list1)

    ax.set_xlabel("p, (error = 1x10^-p)")
    ax.set_ylabel("Iterations")
    ax.set_title("Gauss-Seidel: Convergence test")
    plt.scatter(err_points, gs_iter_list1, label = "Gauss-Seidel", marker = "d")
    plt.plot(err_points, intercept + slope*err_points, 'r', label='Line of best fit')
    plt.plot([], [], label="Gradient = {slope}".format(slope=slope))
    plt.plot([], [], label="Standard error = {std_err}".format(std_err=std_err))
    plt.legend()
    plt.show
    
def J_timer(J_list, jacobi_iter_list2, gs_iter_list2):
    '''measures the iterations required both varing grid dimensions (JxJ). plots
        both methods against their model curves. standard deviation of the data
        shown on plot.'''
    #p is set to be equal to 5 when taking this data (convergence condition is error = 1e-5)
    i = 15
    J_max = 45
    while i<=J_max:
        J_val = i
        J_list.append(J_val)
        
        jacobi_iter_num = jacobi_solution(J_val, J_val, err_total=1e-5, timer=True)
        gs_iter_num = gs_solution(J_val, J_val, err_total=1e-5, timer=True)
        jacobi_iter_list2.append(jacobi_iter_num)
        gs_iter_list2.append(gs_iter_num)
        
        i += 5
    
    #jacobi model curve
    init_guess = [5] #initial guess of the value of p (err = 1e^-p)
    fit = optimize.curve_fit(jacobi_Jmodel, J_list, jacobi_iter_list2, p0=init_guess)  
    ans, cov = fit
    jacobi_p = ans #y values to be plotted on graph
    std_d = np.sqrt(cov) # standard deviation of the data to the model
    
    fig = plt.figure(figsize = (14,8))
    ax = fig.add_subplot(111)
    
    j = np.linspace(15, 45, 5)
    ax.set_xlabel("J, grid dimensions JxJ")
    ax.set_ylabel("Iterations")
    ax.set_title("Jacobi method: time against node number")
    plt.plot(j, jacobi_Jmodel(j, jacobi_p), label="Jacobi model curve")
    plt.scatter(J_list, jacobi_iter_list2, marker = "o")
    plt.plot([], [], label="standard deviation= {std_d}".format(std_d=std_d))
    plt.legend()
    plt.show
    
    #gauss-seidel model curve
    init_guess = [5] # initial guess of the value of p (err = 1e^-p)
    gs_fit = optimize.curve_fit(gs_Jmodel, J_list, gs_iter_list2, p0=init_guess)  
    gs_ans, gs_cov = gs_fit
    gs_p = gs_ans #y values to be plotted on graph
    gs_std_d = np.sqrt(gs_cov) # standard deviation of the data to the model
    
    fig = plt.figure(figsize = (14,8))
    ax = fig.add_subplot(111)
    
    ax.set_xlabel("J, grid dimensions JxJ")
    ax.set_ylabel("Iterations")
    ax.set_title("Gauss-Seidel method: time against node number")
    plt.plot(j, gs_Jmodel(j, gs_p), label="Gauss-Seidel model curve")
    plt.scatter(J_list, gs_iter_list2, marker = "d")
    plt.plot([], [], label="standard deviation= {gs_std_d}".format(gs_std_d=gs_std_d))
    plt.legend()
    plt.show
'-----------------------------------------------------------------------------'
'''Problem 1 show in console'''
#
##grid dimensions for graphs shown in console (not necessarily same in report)
#step_size = 1/128
#num_x = 50
#num_y = 50
#
#print("Calculating the results of the first physics problem...\n")
#print("Potential field solution:\n")
#plot_potential_field_gs(num_x, num_y, step_size)
##plot_vector(num_x, num_y, step_size)
#print("\nElectric field solution:\n")
#plot_electric_field_gs(num_x, num_y, step_size)
#
#print("\n")
#
#err_list1 = []
#err_list2 = []
#J_list = []
#
#gs_iter_list1 = []
#jacobi_iter_list1 = []
#gs_iter_list2 = []
#jacobi_iter_list2 = []
#
#print("\nCalculating iterations required for varying convergence condition...\n")
#jacobi_convergence_timer(err_list1, jacobi_iter_list1)
#gs_convergence_timer(err_list2, gs_iter_list1)
#  
#print("\nCalculating iterations required for varying node number (J)...\n")
#J_timer(J_list, jacobi_iter_list2, gs_iter_list2)
#
#print("(Iteration graphs show in console later)")
'-----------------------------------------------------------------------------'
