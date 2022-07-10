# Import
import math, copy
import numpy as np
import matplotlib.pyplot as plt

# Cost function calculator
def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression 
    Args:
      x (ndarray (m,)): features, m data sets 
      y (ndarray (m,)): target values
      w,b (scalar)    : linear model parameters  
    Returns
      total_cost (scalar): The total cost
     """
   
    # initialize
    m = x.shape[0] 
    cost = 0
    
    # loop over number of data sets
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i])**2
    # total cost
    total_cost = 1 / (2 * m) * cost

    return total_cost

# Cost function gradient calculator
def compute_cost_gradient(x, y, w, b): 
    """
    Computes the gradient of the cost function based on the linear regression model
    Args:
      x (ndarray (m,)): features, m data sets 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost function w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost function w.r.t. the parameter b     
     """
    
    # Initialize
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    
    # loop over number of data sets
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = (f_wb - y[i]) 
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db

# Gradient decent method
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, rel_err): 
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : features, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float)     : learning rate
      num_iters (int)   : number of iterations to run gradient descent
      rel_err(float)    : relative error in the gradient decent
      
    Returns:
      w (scalar)      : Updated value of parameter after running gradient descent
      b (scalar)      : Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
      """
    
    # Initialize
    w = copy.deepcopy(w_in) # avoid modifying global w_in
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b         = b_in
    w         = w_in
    
    # Loop over number of iterations
    i = 0
    while i < num_iters:
        
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = compute_cost_gradient(x, y, w , b)     

        # Update Parameters using equation for the gradient decent
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Save cost J at each iteration
        if i < 100000: # prevent resource exhaustion 
            J_history.append(compute_cost(x, y, w , b))
            p_history.append([w,b])
        # Relative difference
        rel_diff = 1.0
        if i > 0:
            rel_diff = abs(J_history[i]-J_history[i-1])/J_history[i]
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/20) == 0:
            print(f"Itr {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}, rel_err: {rel_diff: 0.5e}")
            
        # Check the convergence and update
        i += 1
        if rel_diff < rel_err:
            return w, b, J_history, p_history
 
    # return w and J,w history for graphing
    return w, b, J_history, p_history

def plot(J_hist):
    # plot cost versus iteration  
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
    ax1.plot(J_hist[:100])
    ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
    ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
    ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
    ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
    plt.show()

    
def main():
    # initialize parameters
    w_init = 0
    b_init = 0

    # some gradient descent settings
    num_iters = 10000
    alpha     = 1.0e-2
    rel_err   = 1.0e-3

    # Load our data set
    x_train = np.array([1.0, 2.0])   #features
    y_train = np.array([300.0, 500.0])   #target value

    # run gradient descent
    w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, alpha, num_iters, rel_err)
    print(f"(J, w, b) found by gradient descent: ({J_hist[-1]:0.7f}, {w_final:8.4f}, {b_final:8.4f})")
    
    # plot
    plot(J_hist)

if __name__ == '__main__':
    main()




