# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 17:10:26 2019

@author: George Flatters
"""

import numpy as np
import scipy
import math
from scipy.stats import chisquare
import time
import matplotlib.pyplot as plt
import pylab
'-----------------------------------------------------------------------------'
'''functions to generate a random nxn matrix and a corresponding vector of known constants'''
n = np.random.randint(2,9) #random matrix dimensions
def get_random_matrix(n):
    '''creates a random matrix of n dimensions, maximum element value of 5'''
    return np.random.randint(9, size=(n, n))

def get_random_vector(n):
    '''creates a random vector g, such that x can be solved for mx = g'''
    return np.random.randint(9, size=(n,1))

m = get_random_matrix(n)
g = get_random_vector(n)
'-----------------------------------------------------------------------------'
'''functions to perform matrix inversion'''

def get_minor(m,i,j):
    '''creates a matrix of minors'''
    return np.delete(np.delete(m,i,0),j,1)

def get_transpose(m):
    '''will be used to transposes the matrix of cofactors'''
    n=len(m)
    empty_array = np.ndarray(shape=(n,n))
    if len(m) == 2:
        return np.array([m[0,0], m[1,0]], [m[0,1], m[1,1]])
    for row in range(len(m)):
        for col in range(len(m[0])):
            empty_array[row][col]=m[col][row]
    return empty_array

def get_det(m):
    '''recursive function that finds the determinant of the matrix'''
    determinant = 0
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0] #special case for 2x2 matrix

    for i in range(len(m)):
        determinant += ((-1)**i)*m[0][i]*get_det(get_minor(m,0,i))
    return determinant

def get_cofactor(m):
    '''finds the matrix of cofactors'''
    get_cofactor = []
    for i in range(len(m)):
        corow = []
        for j in range(len(m)):
            corow.append(((-1)**(i+j)) * get_det(get_minor(m,i,j)))
        get_cofactor.append(corow)
    return np.array(get_cofactor)   

def get_inverse(m):
    if len(m) == 2:
        return np.array([[m[1,1], -m[0,1]], [-m[1,0], m[0,0]]]) * 1/get_det(m)
    '''finds the inverse matrix'''
    for i in range(len(m)):    
        return 1/get_det(m) *  get_transpose(get_cofactor(m))

'-----------------------------------------------------------------------------'
'''methods for solving linear simultaneous equations'''

def get_solution(m, g):
    '''solves using my matrix inversion function'''
    return np.dot(get_inverse(m), g)

def get_lu(m, g):
    '''solves via LU decomposition'''
    lu, piv = scipy.linalg.lu_factor(m) #LU factorises m, to be taken as arguments in lu_solve
    return scipy.linalg.lu_solve((lu, piv), g)

def get_svd(m, g):
    '''solves via SVD'''
    U, S, V = scipy.linalg.svd(m) # factorises m = USV
    c = np.dot(U.T, g) # c = U(trans) * g
    w = np.linalg.solve(np.diag(S), c) #solution of diagonal matrix with c
    return np.dot(V.T, w) #solution x = V*w
    
'-----------------------------------------------------------------------------'
'''tests to see if functions work as intended
   within a tolerance to take rounding errors into account'''

def test_inversion(m):
    '''test to see if matrix inversion is correct'''
    return np.allclose(get_inverse(m), scipy.linalg.inv(m), rtol=1e-05, atol=1e-08, equal_nan=False)

def test_solution(m, g):
    '''solution via matrix inversion test'''
    return np.allclose(get_solution(m, g), scipy.linalg.solve(m, g), rtol=1e-05, atol=1e-08, equal_nan=False)

#def test_accuracy(m, g):
#    return np.dot(get_solution(m), scipy.linalg.inv(m))

def test_lu(m, g):
    '''solution via LU decomposition test'''
    return np.allclose(get_lu(m, g), scipy.linalg.solve(m, g), rtol=1e-05, atol=1e-08, equal_nan=False)

def test_svd(m, g):
    '''solution via SVD test'''
    return np.allclose(get_svd(m, g), scipy.linalg.solve(m, g), rtol=1e-05, atol=1e-08, equal_nan=False)

'-----------------------------------------------------------------------------'
'''functions to test speed of the different methods. against the theoretical fit'''
#set n range for speed tests, upper bounds presented lower than in report to save time
n_min = 2
inv_max = 7
lu_max = 151
svd_max = 101

def inv_fit(n):
    '''The cost of solving a system of linear equations is approximately
    n*n! floating-point operations for matrix of n dimensions '''
    flops_inv = 5e4
    return 50*(n*math.factorial(n))/flops_inv#50 repeats

def inv_time(inv_n, inv_total, inv_fitdata):
    '''measures the speed of the lud method for a given number of repeats 
        for varying values of n'''
    for N in range(n_min,inv_max): #leave this as (2,6) for hand in
        n = N
        inv_n.append(n)
        theory = inv_fit(n)
        inv_fitdata.append(theory)
        
        repeat = 50
        f=0
        while f<repeat:
            m = get_random_matrix(n) 
            g = get_random_vector(n)
            
            if np.linalg.det(m) == 0: #if the matrix is singular do not run
                i = 0
                
            else:
                start = time.time()
                i = get_solution(m,g) #waste variable to run once
                end = time.time()
                inv_list.append(end-start)
                f+=1#gives same number of non singular matrices
        inv_total.append(np.sum(inv_list))#sums all the times together
    return inv_n, inv_total, inv_fitdata


def lu_fit(n, a, b, c):
    '''The cost of solving a system of linear equations is approximately
    (2/3)*n**3 floating-point operations for matrix of n dimensions '''
    flops_lu = 5e7
    return 150*((a/b)*n**c)/flops_lu #time taken for 150 repeats of lu decompositions for n

def lu_time(lu_n,lu_total, lu_fitdata):
    '''measures the speed of the lud method for a given number of repeats 
        for varying values of n'''
    for N in range(n_min,lu_max): #leave this as (2,6) for hand in
        n = N
        lu_n.append(n)
        a = 2
        b = 3
        c = 3
        theory = lu_fit(n, a, b, c)
        lu_fitdata.append(theory)
        
        repeat = 150
        f=0
        while f<repeat:
            m = get_random_matrix(n) 
            g = get_random_vector(n)
            
            if np.linalg.det(m) == 0: #if the matrix is singular do not run
                i = 0
                
            else:
                start = time.time()
                i = get_lu(m,g) #waste variable to run once
                end = time.time()
                lu_list.append(end-start)
                f+=1#gives same number of non singular matrices
        lu_total.append(np.sum(lu_list))#sums all the times together
    return lu_n, lu_total, lu_fitdata

def svd_fit(n, a):
    '''The cost of solving a system of linear equations is approximately
    n**3 floating-point operations for matrix of n dimensions '''
    flops_svd = 8e6
    return 100*(n**a)/flops_svd

def svd_time(svd_n, svd_total, svd_fitdata):
    '''measures the speed of the lud method for a given number of repeats 
        for varying values of n'''
    for N in range(n_min,svd_max): #leave this as (2,6) for hand in
        n = N
        svd_n.append(n)
        a = 3
        theory = svd_fit(n, a)
        svd_fitdata.append(theory)
        
        repeat = 100
        f=0
        while f<repeat:
            m = get_random_matrix(n) 
            g = get_random_vector(n)
            
            if np.linalg.det(m) == 0: #if the matrix is singular do not run
                i = 0
                
            else:
                start = time.time()
                i = get_svd(m,g) #waste variable to run once
                end = time.time()
                svd_list.append(end-start)
                f+=1#gives same number of non singular matrices
        svd_total.append(np.sum(svd_list))#sums all the times together
    return svd_n, svd_total, svd_fitdata

'-----------------------------------------------------------------------------'
'''function for testing behaviour when equations approach singular'''

def k_accuracy(k_list, k_error, scipy_error):
    '''finds the difference between the identity matrix and the matrix produced by
        multiplying the inverse of m by m. this gives the error produced by my inversion function.
        this is compared to the scipy inversion'''
    for K in np.arange(1e-15,1e-13,1e-16):#would become singular for k less than 1e-15
        k=K
        k_list.append(k)
        m = np.array([[1,1,1],[1,2,-1],[2,3,k]])
        
        acc = np.dot(m,get_inverse(m))#finds identity matrix using my routine
        dif = abs(np.subtract(np.identity(3),acc))#compared with actual I
        
        scipy_acc = np.dot(m, scipy.linalg.inv(m)) #finds identity matrix using scipy.inv
        scipy_dif = -abs(np.subtract(np.identity(3), scipy_acc))#scipy identity matrix compared with I
        
        total = np.sum(dif)
        scipy_total = np.sum(scipy_dif)
        
        k_error.append(total)
        scipy_error.append(scipy_total)
    return k_list, k_error, scipy_error
'-----------------------------------------------------------------------------'
'''functions for the 2d physics problem'''

def get_position2d(x,y):
    '''gives the position matrix 2d'''
    A = np.zeros((2,2))
    A[0][0] = x/np.sqrt(x**2 + (8-y)**2)
    A[0][1] = (x - 15)/np.sqrt((15-x)**2 + (8-y)**2)
    A[1][0] = (8-y)/np.sqrt(x**2 + (8-y)**2)
    A[1][1] = (8-y)/np.sqrt((15-x)**2 + (8-y)**2)
    
    return A

@np.vectorize #takes np.arrays as inputs and returns a single np.array
def get_tension2d(x,y):
    '''finds the tension in the first string'''
    lu, piv = scipy.linalg.lu_factor(get_position2d(x,y)) 
    T = scipy.linalg.lu_solve((lu,piv), F)
    return T[0] #returns the tension in the front left string

def get_xy(x, y):
    '''finds x and y for the maximum tension using the index of the unraveled tension grid'''
    index = np.unravel_index(np.nanargmax(t1, axis=None), t1.shape) #finds index of x,y
    x_max = x[index]
    y_max = y[index]
    return x_max, y_max
'-----------------------------------------------------------------------------'
'''functions for the 3d physics problem'''
def get_position3d(x1,y1,z1):
    '''gives the position matrix 3d. here z is going back into the stage and
       y is the verical axis'''
    A = np.zeros((3,3))
    n1 = np.sqrt(z1**2 + (15-x1)**2)
    n2 = np.sqrt(z1**2 + (x1)**2)
    n3 = np.sqrt((8-z1)**2 + (7.5-x1)**2)
    
    A[0][0] = (8-y1)/ np.sqrt(n2**2 +(8-y1)**2)
    A[0][1] = (8-y1)/ np.sqrt(n1**2 +(8-y1)**2)
    A[0][2] = (8-y1)/ np.sqrt(n3**2 +(8-y1)**2)
    A[1][0] = -x1 / np.sqrt(n2**2 +(8-y1)**2)
    A[1][1] = (15-x1)/ np.sqrt(n1**2 +(8-y1)**2)
    if x1<7.5:
        A[1][2] = (7.5-x1) / np.sqrt(n3**2 +(8-y1)**2)
    else:
        A[1][2] = (x1-7.5) / np.sqrt(n3**2 +(8-y1)**2)
    A[2][0] = -z1/ np.sqrt(n2**2 +(8-y1)**2)
    A[2][1] = -z1/ np.sqrt(n1**2 +(8-y1)**2)
    A[2][2] = (8-z1)/ np.sqrt(n3**2 +(8-y1)**2)
    return A    

@np.vectorize #takes np.arrays as inputs and returns a single np.array
def get_tension3d(x1, y1, z1):
    '''finds the tension in the first string'''
    lu, piv = scipy.linalg.lu_factor(get_position3d(x1, y1, z1)) 
    T = scipy.linalg.lu_solve((lu,piv), F_3d)
    return T[0]

def get_xyz(x1, y1, z1):
    '''finds x, y and for the maximum tension using the index of the unraveled tension grid'''
    index = np.unravel_index(np.nanargmax(t1_3d, axis=None), t1_3d.shape) #max tension index
    x_max = x1[index] #gets x value
    y_max = y1[index] #gets y value
    z_max = z1[index] #gets z value
    return x_max, y_max, z_max
'-----------------------------------------------------------------------------'
'''prints the result of the matrix inversion and the solutions to the
   matrix equation mx = g via the three methods'''

print('The matrix is: \n', m)
print('The vector is: \n', g)

#inverse matrix result
print ('Correct inversion?', test_inversion(m), '\n')
print('The inverse matrix is:\n',get_inverse(m), '\n')

#analytic solution
print('Correct solution via analytic method?', test_solution(m, g), '\n')
print('Solution via matrix inversion: \n', get_solution(m, g), '\n')

#analytic solution accuracy
#print('How accurate is the analytic method?', test_accuracy(), '\n')

#LU decomposition solution
print ('Correct solution via LU decomposition?', test_lu(m ,g), '\n')
print ('Solution via LU decomposition: \n', get_lu(m, g), '\n')

#SVD solution
print('Correct solution via SVD?', test_svd(m, g), '\n')
print ('Solution via SVD: \n', get_svd(m, g), '\n')
'-----------------------------------------------------------------------------'
'''speed tests, some hashed out to save time'''
print('Please allow some time for the speed graphs to plot...')
#different lists:

inv_list = []
svd_list = []
lu_list = []

#lists for n data
inv_n = []
lu_n = []
svd_n = []

#lists for time data
inv_total = []
svd_total = []
lu_total = []

#lists for fit data
lu_fitdata = []
inv_fitdata = []
svd_fitdata = []

inv_n, inv_total, inv_fitdata = inv_time(inv_n, inv_total, inv_fitdata)
lu_n, lu_total, lu_fitdata = lu_time(lu_n, lu_total, lu_fitdata)
svd_n, svd_total, svd_fitdata = svd_time(svd_n, svd_total, svd_fitdata)

#'''analytic method speed test'''
#fig = plt.figure(figsize = (18,10))
#ax = fig.add_subplot(111)
#ax.plot(inv_n, inv_total, c='blue')
#ax.set_xlabel("n")
#ax.set_ylabel("Time (Seconds)")
#ax.set_xlim(n_min, inv_max-1)
#ax.set_ylim(0)
#ax.set_title('Speed of analytic method')
#ax.grid()
#plt.show()

#'''Algorithm speed tests'''
#
#fig = plt.figure(figsize = (18,10))
#ax = fig.add_subplot(111)
#ax.plot(lu_n, lu_total, c='magenta', label='LUD')
#ax.plot(svd_n, svd_total, c='yellow', label='SVD')
#ax.set_xlabel("n")
#ax.set_ylabel("Time (Seconds)")
#ax.set_xlim(n_min, lu_max-1)
#ax.set_ylim(0)
#ax.set_title('Speed of the two algorithms')
#ax.grid()
#pylab.legend(prop={'size':15})
#plt.show()

#'''SVD speed test'''
#fig = plt.figure(figsize = (18,10))
#ax = fig.add_subplot(111)
#ax.plot(svd_n, svd_total, c='yellow')
#ax.set_xlabel("n")
#ax.set_ylabel("Time (Seconds)")
#ax.set_xlim(n_min, svd_max-1)
#ax.set_ylim(0)
#ax.set_title('Speed of singular value decomposition algorithm')
#ax.grid()
#plt.show()

#'''all speed tests'''
#fig = plt.figure(figsize = (18,10))
#ax = fig.add_subplot(111)
#ax.plot(inv_n, inv_total, c='blue', label='Analytic Method')
#ax.plot(lu_n, lu_total, c='magenta', label='LUD')
#ax.plot(svd_n, svd_total, c='yellow', label='SVD')
#ax.set_xlabel("n")
#ax.set_ylabel("Time (Seconds)")
#ax.set_xlim(n_min, inv_max-1)
#ax.set_ylim(0)
#ax.set_title('Speeds of all methods compared')
#ax.grid()
#pylab.legend(prop={'size':15})
#plt.show()
'-----------------------------------------------------------------------------'
'''statistical test for analytic method'''
fig = plt.figure(figsize = (18,10))
ax = fig.add_subplot(111)
ax.plot(inv_n, inv_total, c='magenta', label='Analytic method time')
ax.plot(inv_n, inv_fitdata, 'g--', label='Theoretical fit')
ax.set_xlabel("n")
ax.set_ylabel("Time (Seconds)")
ax.set_xlim(n_min, inv_max-1)
ax.set_ylim(0)
ax.set_title('Speed of Analytic Method')
ax.grid()
pylab.legend(prop={'size':15})
plt.show()

print('Results of the analytic method chi-square test:')
print(chisquare([inv_total], f_exp=[inv_fitdata], ddof=5, axis=None))

'''statistical test for lud'''

fig = plt.figure(figsize = (18,10))
ax = fig.add_subplot(111)
ax.plot(lu_n, lu_total, c='magenta', label='LU decomposition time')
ax.plot(lu_n, lu_fitdata, 'r--', label='FLOP model fit')
ax.set_xlabel("n")
ax.set_ylabel("Time (Seconds)")
ax.set_xlim(n_min, lu_max-1)
ax.set_ylim(0)
ax.set_title('Speed of LUD')
ax.grid()
pylab.legend(prop={'size':15})
plt.show()

print('Results of the LU chi-square test:')
print(chisquare([lu_total], f_exp=[lu_fitdata], ddof=148, axis=None))

'''statistical test for svd'''
fig = plt.figure(figsize = (18,10))
ax = fig.add_subplot(111)
ax.plot(svd_n, svd_total, c='magenta', label='Singular value decomposition time')
ax.plot(svd_n, svd_fitdata, 'g--', label='theoretical fit')
ax.set_xlabel("n")
ax.set_ylabel("Time (Seconds)")
ax.set_xlim(n_min, svd_max-1)
ax.set_ylim(0)
ax.set_title('Speed of SVD')
ax.grid()
pylab.legend(prop={'size':15})
plt.show()

print('Results of the SVD chi-square test:')
print(chisquare([svd_total], f_exp=[svd_fitdata], ddof=98, axis=None))
'-----------------------------------------------------------------------------'
'''behaviour when equations approach singular'''
k_list = []
k_error = []
scipy_error = []

k_list, k_error, scipy_error = k_accuracy(k_list, k_error, scipy_error)

fig = plt.figure(figsize = (18,10))
ax = fig.add_subplot(111)
ax.plot(k_list, k_error, c='blue', label='my inversion error')
ax.plot(k_list, scipy_error, c='green', label='scipy inversion error')
ax.set_xlabel("k")
ax.set_ylabel("absolute k error")
ax.set_xlim(1e-15, 1e-13)
ax.set_title('The error in the inversion as a function of k')
ax.grid()
pylab.legend(prop={'size':15})
plt.show()

'-----------------------------------------------------------------------------'
'''results of the 2d physics problem'''

F = np.array([[0], [70*9.81]]) #force vector 2d
points_2d = 200 #number of points for 2d plots

x,y = np.meshgrid([np.linspace(0, 15, points_2d)], [np.linspace(0, 7, points_2d)])
t1 = get_tension2d(x, y) #tension values on meshgrid

fig, ax = plt.subplots(figsize = (18,10))
c = ax.pcolormesh(x, y, t1, cmap='RdBu')

ax.set_title('2D Tension Graph')
ax.set_ylabel('y')
ax.set_xlabel('x')
ax.set_xlim(0, 15)
ax.set_ylim(0, 7)
fig.colorbar(c, ax=ax, label='Tension (N)')
plt.show()

print('The maximum tension (2d) of the left string is: \n', np.amax(t1))
print('The tension (2d) is greatest at: \n', get_xy(x, y))
'-----------------------------------------------------------------------------'
'''results of the 3d physics problem'''

F_3d = np.array([[0], [0], [-700]]) #force vector 3d
points_3d = 50 #number of points for 3d plots

x1, y1 , z1 = np.meshgrid([np.linspace(0, 8, points_3d)], [np.linspace(0, 15, points_3d)], [np.linspace(0, 7, points_3d)])
t1_3d = get_tension3d(x1, y1, z1) #tension values on 3d meshgrid

print('The maximum tension (3d) of the left string is: \n', np.amax(t1_3d))
print('The tension (3d) is greatest at: \n', get_xyz(x1, y1, z1))


       