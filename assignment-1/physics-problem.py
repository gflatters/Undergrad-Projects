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
