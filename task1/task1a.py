import numpy as np
import torch 
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision

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

def fit_polynomial_sgd_weight_regularised(x,t,M, lr, batch_size):
    """ Similar to fit_polynomial_sgd but with an added penalty term to the loss dependant on the L2-norm of the weight vector to encourage small weights.
        INPUT: vectorised scalars in x(shape number of data points by 1), target values t(shape number of data points by 1), learning rate lr, and batch size. M here represents the maximum degree of the class of polynomials we're optimising over
        OUTPUT: optimum weight vector 𝐰̂(shape 1 by M+1) found by applying sgd to minimise squared distance of t and y=𝐰̂x while also trying to minimise model complexity(in the form of polynomial degree)
     """
    #print message
    print(". \n"*5)
    print("-" * 20 + "start" + "-" * 20)
    print("Starting SGD training to find optimum \nweight vector (optimising over polynomial degree as well). \nWill train for 150_000 epochs. \nMaximum polynomial degree considered is " + str(M))
    
    
    #scaling inputs to avoid exploding gradients
    powers_of_x = torch.pow(x, torch.arange(M+1, dtype=torch.float32))
    maximum_powers_of_x = (torch.max(torch.abs(powers_of_x), axis=0)).values
    powers_of_x = torch.div(powers_of_x, maximum_powers_of_x)
    data_set = TensorDataset(powers_of_x,t)
    

    #simple linear layer where the weights get multiplied by the powers of x and we optimise these weights to find the polynomial fit
    model = nn.Linear(M+1,1, bias=False, dtype=torch.float32)
    loss_func = nn.MSELoss()

    optimiser = torch.optim.SGD(model.parameters(), lr=lr)
    
    alpha = 10**(-M) #weight regularisation alpha
    for epoch in range(150_000):
        #batches
        data_batches = DataLoader(data_set, batch_size=batch_size, shuffle=True)
        for input, ground_truth in data_batches:
            optimiser.zero_grad() #emptying gradients to avoid accumulation
            prediction = model(input) #prediction step

            regularisation_term = torch.sum(torch.square(torch.div(model.weight, maximum_powers_of_x))) #regularisation with L2-norm
            loss = loss_func(prediction, ground_truth) + alpha*regularisation_term #loss with weight regularisation
            loss.backward() #derivative of loss
            optimiser.step() #optimisation

            


        if(epoch%10_000==0):
            #every 10_000 epochs setting weights that are not very influential (when considering the degree they represent) to 0
            flag = torch.abs(torch.div(model.weight, maximum_powers_of_x)) >= 1e-3
            model.weight.data = model.weight*flag
            print("Epoch: " + str(epoch) + ", MSE + weight L2 regularisation loss: " + str((loss).tolist()))

    print("Epoch: " + str(epoch) + ", MSE + weight L2 regularisation loss: " + str((loss).tolist()))
    
    flag = torch.abs(torch.div(model.weight, maximum_powers_of_x)) >= 1e-3
    model.weight.data = model.weight*flag
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

    #Report, using printed messages, the optimised 𝑀 value and the mean (and standard deviation) in 
    #difference between the model-predicted values and the underlying “true” polynomial curve.  
    M_max = 10 #maximum polynomial degree allowed during fitting
    w_hat_sgd = fit_polynomial_sgd_weight_regularised(x_train, t_train, M_max, 0.5, 25) #batch_size=25, learning rate = 0.5
    print("Optimised weight vector is: " + str(w_hat_sgd.tolist()))
    optimal_M = torch.max(torch.nonzero(w_hat_sgd)[:,0]) #optimal polynomial degree after training
    print("Thus the optimised degree of the polynomial is: " + str(optimal_M.item()))

    y_hat_sgd_train = polynomial_fun(w_hat_sgd, x_train)
    y_hat_sgd_test = polynomial_fun(w_hat_sgd, x_test)

    #on training set
    difference = y_hat_sgd_train - y_train
    std_difference, mean_difference, = torch.std_mean(difference)
    print(". \n"*5)
    print("-" * 20 + "start" + "-" * 20)
    print("Difference between \npredicted values (on training set) \nand true polynomial")
    print("Mean : " + str(mean_difference.tolist()))
    print("STD : " + str(std_difference.tolist()))
    print("-" * 20 + "end" + "-" * 20)
    del difference, std_difference, mean_difference


    #on testing set
    difference = y_hat_sgd_test - y_test
    std_difference, mean_difference, = torch.std_mean(difference)
    print(". \n"*5)
    print("-" * 20 + "start" + "-" * 20)
    print("Difference between \npredicted values (on testing set) \nand true polynomial")
    print("Mean : " + str(mean_difference.tolist()))
    print("STD : " + str(std_difference.tolist()))
    print("-" * 20 + "end" + "-" * 20)
    del difference, std_difference, mean_difference


if __name__=="__main__":
    main()
