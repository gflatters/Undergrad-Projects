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
