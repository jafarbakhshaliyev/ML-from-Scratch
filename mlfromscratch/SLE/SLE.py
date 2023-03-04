import numpy as np

def solve_SLE(A, b, solver= "Gaussian"):
    """
    The function solves linear system of equations by taking A and b where Ax = b and solver: Gaussian or QR 
    and returns.
    :param A: array
    :param b: array
    :param solver: str ("Gaussian" or "QR")
    """
    if solver == 'Gaussian':

        A, b = Gaussian_elimination(A, b) 
        x = back_substition(A,b)

    elif solver == 'QR':

        A, b = QR_decomposition(A,b)  
        x = back_substition(A,b)

    else:
        raise Exception('Please enter correct method name as Gaussian or QR')
    
    return x

def Gaussian_elimination(A, b):
     
    for k in range(len(b)-1): 

        if A[k,k] == 0: raise Exception('Math Error') 

        for i in range(k+1,len(b)): 

            factor = A[i,k]/A[k,k] 

            for j in range(k,len(b)): 

                A[i,j] -= factor*A[k,j] 
            b[i] = np.subtract(b[i], np.float64(factor), casting='unsafe')

    return A, b             
    

def QR_decomposition(A, b):
    
    Q = np.zeros((A.shape[0],A.shape[0])) 
    o = np.zeros((A.shape[0],A.shape[0])) 
    o[:,0] = A[:,0] 
    Q[:,0] = o[:,0] / np.linalg.norm(o[:,0])

    for i in range(1, A.shape[0]):

        o[:,i] = A[:,i] 
        for j in range(i):

            o[:, i] -= np.dot(A[:, i], Q[:, j]) * Q[:, j] 
                                                      
        Q[:,i] = o[:, i]/np.linalg.norm(o[:,i]) 
        
    R = np.zeros((A.shape[0],A.shape[1]))
    
    for i in range(A.shape[0]): 

        for j in range(i, A.shape[1]):   

            R[i,j] = np.dot(A[:,j], Q[:,i])
                                           
    return R, np.dot(Q.T,b) 

def back_substition(A, b):

    x = np.zeros(len(b), dtype = float) 
    x[len(b)-1] = b[len(b)-1]/A[len(b)-1,len(b)-1] 

    for i in range(len(b) - 2, -1, -1): 

        total = b[i] 

        for j in range(i+1,len(b)): 

            #total -= A[i,j]*x[j] 
            total = total - np.subtract(total, np.float64(A[i,j]*x[j] ), casting='unsafe')
                                
        x[i] = total/A[i,i] 

    return x  
