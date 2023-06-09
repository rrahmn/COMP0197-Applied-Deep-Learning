import numpy as np
import torch 
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import time

def polynomial_fun(w, x):
    """ Implement a polynomial function polynomial_fun, that takes two input arguments, a weight vector 𝐰 
        of size 𝑀+1 and an input scalar variable 𝑥, and returns the function value 𝑦.  The polynomial_fun 
        should be vectorised for multiple pairs of scalar input and output, with the same 𝐰

        INPUT: weight vector w(shape 1 by M+1) and vectorised scalars in x(shape number of data points by 1)
        OUPUT: polynomial evaluation at each scalar value in x using w entries as coefficients. Returned as vector y(shape number of data points by 1).
    """
    size = w.shape[0]
    powers_of_x = torch.pow(x, torch.arange(size, dtype=torch.float32))
    y = torch.matmul(powers_of_x, w)
    return y

def fit_polynomial_ls(x,t,M):
    """ Using the linear algebra modules in PyTorch, implement a least square solver for fitting 
        the polynomial functions, fit_polynomial_ls, which takes 𝑁 pairs of 𝑥 and target values 𝑡 as input, with 
        an additional input argument to specify the polynomial degree 𝑀, and returns the optimum weight 
        vector 𝐰̂ in least-square sense, i.e. ‖𝑡−𝑦‖2 is minimised.
        INPUT: vectorised scalars in x(shape number of data points by 1), target values t(shape number of data points by 1), polynomial degree M
        OUTPUT: optimum weight vector 𝐰̂(shape 1 by M+1) minimising squared distance of t and y=𝐰̂x
    """
    #scaling inputs to avoid exploding powers
    powers_of_x = torch.pow(x, torch.arange(M+1, dtype=torch.float32))
    maximum_powers_of_x = (torch.max(torch.abs(powers_of_x), axis=0)).values
    powers_of_x = torch.div(powers_of_x, maximum_powers_of_x)
    w_hat = torch.linalg.lstsq(powers_of_x, t).solution
    w_hat = (torch.div(w_hat.T, maximum_powers_of_x)).T  #rescaling to invert effect of initial scaling
    return w_hat

def fit_polynomial_sgd(x,t,M, lr, batch_size):
    """ This function also returns the optimum weight vector by implementing a 
        stochastic minibatch gradient
        descent algorithm for fitting the polynomial functions
        INPUT: vectorised scalars in x(shape number of data points by 1), target values t(shape number of data points by 1), polynomial degree M, learning rate lr, and batch size
        OUTPUT: optimum weight vector 𝐰̂(shape 1 by M+1) found by applying sgd to minimise squared distance of t and y=𝐰̂x
     """
    #print message
    print(". \n"*5)
    print("-" * 20 + "start" + "-" * 20)
    print("Starting SGD training to find optimum \nweight vector. Will train for 10_000 epochs.")
    
    
    #scaling inputs to avoid exploding gradients
    powers_of_x = torch.pow(x, torch.arange(M+1, dtype=torch.float32))
    maximum_powers_of_x = (torch.max(torch.abs(powers_of_x), axis=0)).values
    powers_of_x = torch.div(powers_of_x, maximum_powers_of_x)
    data_set = TensorDataset(powers_of_x,t)
    

    #simple linear layer where the weights get multiplied by the powers of x and we optimise these weights to find the polynomial fit
    model = nn.Linear(M+1,1, bias=False, dtype=torch.float32)
    loss_func = nn.MSELoss() #mean squared error as loss

    optimiser = torch.optim.SGD(model.parameters(), lr=lr) #standard stochastic gradient descent
    
    for epoch in range(10_000):
        #batches
        data_batches = DataLoader(data_set, batch_size=batch_size, shuffle=True)
        for input, ground_truth in data_batches:
            optimiser.zero_grad() #emptying gradients to avoid accumulation
            prediction = model(input) #prediction step
            loss = loss_func(prediction, ground_truth) #loss 
            loss.backward() #derivative of loss
            optimiser.step() #optimisation

        if(epoch%1_000==0):
            print("Epoch: " + str(epoch) + ", MSE loss: " + str((loss).tolist()))
    
    print("Epoch: " + str(epoch) + ", MSE loss: " + str((loss).tolist()))
    
    w_hat = model.weight
    #we scaled the data by a scale factor of maximum_powers_of_x at the beginning causing the weights to be divided by maximum_powers_of_x so now we need to scale the weight
    print("-" * 20 + "end" + "-" * 20)
    w_hat = (torch.div(w_hat, maximum_powers_of_x)).T #transpose to make consisstent with convention of polynomial_fun(w, x)
    return w_hat




def main():


    #Use polynomial_fun (𝑀 =4, 𝐰=[1,2,3,4,5]T) to generate a training set and a test set, in the 
    #form of respectively sampled 100 and 50 pairs of 𝑥,𝑥𝜖[−20,20], and 𝑡. The observed 𝑡 values 
    #are obtained by adding Gaussian noise (standard deviation being 0.2) to 𝑦.
    
    temp1 = torch.arange(5, dtype=torch.float32)
    temp2 = torch.tensor(1, dtype=torch.float32)
    w = torch.add(temp1, temp2)
    w = w.reshape(w.shape[0], -1)
    del temp1, temp2
    w = torch.tensor([1,2,3,4,5], dtype=torch.float32).reshape(5,1)

    

    #training set
    x_train = 40.0*(torch.rand(100, dtype=torch.float32) - 0.5).reshape(100,1)
    y_train = polynomial_fun(w,x_train)
    noise_train = (0.2*torch.randn(100, dtype=torch.float32)).reshape(100,1)
    t_train = y_train+noise_train
    del noise_train


    #testing set
    x_test = 40.0*(torch.rand(50, dtype=torch.float32) - 0.5).reshape(50,1)
    y_test = polynomial_fun(w,x_test)
    noise_test = 0.2*torch.randn(50, dtype=torch.float32).reshape(50,1)
    t_test = y_test + noise_test
    del noise_test

    

    #Use fit_polynomial_ls (𝑀 =5) to compute the optimum weight vector 𝐰̂ using the training 
    #set. In turn, compute the predicted target values 𝑦̂ for all 𝑥 in both the training and test sets. 
    time_ls = time.time()
    w_hat_ls = fit_polynomial_ls(x_train,t_train,M=5)
    time_ls = time.time() - time_ls #time spent fitting for ls
    y_hat_ls_train = polynomial_fun(w_hat_ls,x_train)
    y_hat_ls_test = polynomial_fun(w_hat_ls,x_test)

    #Report, using printed messages, the mean (and standard deviation) in difference a) between 
    #the observed training data and the underlying “true” polynomial curve; and b) between the 
    #“LS-predicted” values and the underlying “true” polynomial curve.

    #a)
    #TODO difference or absolute difference?
    difference = t_train - y_train
    std_difference, mean_difference, = torch.std_mean(difference)
    print(". \n"*5)
    print("-" * 20 + "start" + "-" * 20)
    print("Difference between \nobserved training data \nand true polynomial")
    print("Mean : " + str(mean_difference.tolist()))
    print("STD : " + str(std_difference.tolist()))
    print("-" * 20 + "end" + "-" * 20)
    del difference, std_difference, mean_difference
    
    #b)
    difference = y_hat_ls_train - y_train
    std_difference, mean_difference, = torch.std_mean(difference)
    print(". \n"*5)
    print("-" * 20 + "start" + "-" * 20)
    print("Difference between \nLS-predicted values (on training set) \nand true polynomial")
    print("Mean : " + str(mean_difference.tolist()))
    print("STD : " + str(std_difference.tolist()))
    print("-" * 20 + "end" + "-" * 20)
    del difference, std_difference, mean_difference
    

    #Use fit_polynomial_sgd (𝑀 = 5) to optimise the weight vector 𝐰̂ using the training set. In
    #turn, compute the predicted target values 𝑦̂ for all 𝑥 in both the training and test sets.
    
    time_sgd = time.time()
    w_hat_sgd = fit_polynomial_sgd(x_train, t_train, 5, 0.25, 25) #batch_size=10, learning rate = 0.25
    time_sgd = time.time() - time_sgd #time spent training for sgd

    y_hat_sgd_train = polynomial_fun(w_hat_sgd, x_train)
    y_hat_sgd_test = polynomial_fun(w_hat_sgd, x_test)
    
    #Report, using printed messages, the mean (and standard deviation) in difference between the 
    #“SGD-predicted” values and the underlying “true” polynomial curve.

    #on training set
    difference = y_hat_sgd_train - y_train
    std_difference, mean_difference, = torch.std_mean(difference)
    print(". \n"*5)
    print("-" * 20 + "start" + "-" * 20)
    print("Difference between \nSGD-predicted values (on training set) \nand true polynomial")
    print("Mean : " + str(mean_difference.tolist()))
    print("STD : " + str(std_difference.tolist()))
    print("-" * 20 + "end" + "-" * 20)
    del difference, std_difference, mean_difference


    #Compare the accuracy of your implementation using the two methods with ground-truth on 
    #test  set  and  report  the  root-mean-square-errors  (RMSEs)  in  both  𝐰  and  𝑦  using  printed 
    #messages.
    
    mse_ls = torch.square(y_hat_ls_test - t_test)
    std_mse_ls, mean_mse_ls, = torch.std_mean(mse_ls)

    mse_sgd = torch.square(y_hat_sgd_test - t_test)
    std_mse_sgd, mean_mse_sgd, = torch.std_mean(mse_sgd)
    padding = nn.ZeroPad2d((0,0,0,1)) #padding with 0 at the end to make sure we can find rmse of w (since original was size 5 and w_hat are size 6)
    rmse_w_ls = torch.sqrt(torch.mean(torch.square(w_hat_ls - padding(w))))
    rmse_w_sgd = torch.sqrt(torch.mean(torch.square(w_hat_sgd - padding(w))))
    rmse_y_ls = torch.sqrt(torch.mean(torch.square(y_hat_ls_test - y_test)))
    rmse_y_sgd = torch.sqrt(torch.mean(torch.square(y_hat_sgd_test - y_test)))

    print(". \n"*5)
    print("-" * 40 + "start" + "-" * 40)
    print("MSE between predicted values \n(on testing set) and ground truth values t\n")
    print(f"{'LS' : <50}{'SGD' : <30}")
    print(f"{'Mean: ' + str(mean_mse_ls.tolist())  : <50}{'Mean: ' + str(mean_mse_sgd.tolist()): <50}")
    print(f"{'STD: ' + str(std_mse_ls.tolist())  : <50}{'STD: ' + str(std_mse_sgd.tolist()): <50}")
    print(f"{'RMSE for w: ' + str(rmse_w_ls.tolist())  : <50}{'RMSE for w: ' + str(rmse_w_sgd.tolist()): <50}")
    print(f"{'RMSE for y(test): ' + str(rmse_y_ls.tolist())  : <50}{'RMSE for y(test): ' + str(rmse_y_sgd.tolist()): <50}")
    print(f"{'Training time(seconds) : ' + str(time_ls)  : <50}{'Training time(seconds) : ' + str(time_sgd): <50}")
    print("As we can see SGD arrives at a more accurate solution (smaller mean mse) and it \nalso has a smaller std indicating the errors across the data are similar and well spread. \nFurthermore, the rmse for w and y are much smaller for SGD than LS, so SGD fits the \npolynomial better and generalises better too. However, SGD has a much longer training time.")
    print("-" * 40 + "end" + "-" * 40)

    del mse_ls, std_mse_ls, mean_mse_ls
    del mse_sgd, std_mse_sgd, mean_mse_sgd
    



if __name__=="__main__":
    main()
