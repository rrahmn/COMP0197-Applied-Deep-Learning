import torch
import numpy as np

class mixup:
    """ mixup algorithm as described in paper """
    def __init__(self, alpha, sampling_method):
        self.alpha = alpha
        self.sampling_method = sampling_method

    def mix(self, x, y):
        """ INPUT: vectors to be mixed x and y of shape batch_size by 3 by image height by image weidth and batch_size by 10 respectively along with sampling method flag
            OUTPUT: mixed vectors mixed_x, mixed_y of same shape. data type either torch.float32 or torch.FloatTensor depending on cpu or gpu usage """
        #sample lambda depending on sampling_method
        if(self.sampling_method==1):
            l = np.random.beta(self.alpha,self.alpha)
        if(self.sampling_method==2):
            l = np.random.uniform(0,0.5) #range 0 to 0.5

        indices = torch.randperm(x.shape[0]).to(x.device)
        mixed_x = l*x + (1 - l)*x[indices,:,:,:]
        mixed_y = l*y + (1 - l)*y[indices]

        return mixed_x, mixed_y