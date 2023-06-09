import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image

import numpy as np
from resnet50_network_pt import MyResnet50 #resnet50
from mixup_class import mixup
import time


def main():
    print("-" * 20 + "start" + "-" * 20)
    print("Chosen modification: Difference between using SGD with momentum (as in “train_pt.py”) and Adam optimiser (as in “train_tf.py”), with sampling_method = 1")
    print("-" * 20 + "end" + "-" * 20)

    
    #checking for gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if(torch.cuda.is_available()):
        torch.cuda.set_device(device)
        torch.set_default_tensor_type(torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor)
    device = 'cpu'

    generator = torch.Generator(device) #initialising generator on right device
    generator.manual_seed(np.random.randint(0,1000)) #different seed every run

    # cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    #Split the data into development set (80%) and holdout test set (20%)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    entire_dataset = torch.utils.data.ConcatDataset([trainset,testset])
    del trainset, testset


    length_dataset = len(entire_dataset)
    development_set_size = int(0.8*length_dataset)
    holdout_set_size = length_dataset - development_set_size

    development_set, holdout_set = torch.utils.data.random_split(entire_dataset,[development_set_size, holdout_set_size], generator=generator)
    del entire_dataset, length_dataset

    #Random-split the development set in the train (90%) and validation sets (10%)
    train_set_size = int(0.9*development_set_size)
    validation_set_size = development_set_size - train_set_size

    train_set, validation_set = torch.utils.data.random_split(development_set,[train_set_size, validation_set_size], generator=generator)
    del development_set, development_set_size

    
    #Train two models using the different modifications (λ sampling methods or optimisers). 
    
    #training
    batch_size = 20
    #data loading
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, generator=generator)
    validationloader = torch.utils.data.DataLoader(validation_set, batch_size=validation_set_size, shuffle=True, num_workers=2, generator=generator)
    testloader = torch.utils.data.DataLoader(holdout_set, batch_size=holdout_set_size, shuffle=True, num_workers=2, generator=generator)
    
    
    net1 = MyResnet50() #for investigating performance before modification
    net2 = MyResnet50() #for investigating performance with sgd optimiser
    net3 = MyResnet50() #for investigating performance with adam optimiser
    net1.to(device) #making sure we're on right device
    net2.to(device) #making sure we're on right device
    net3.to(device) #making sure we're on right device

    # loss and optimiser
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss() #switched to MSE instead of cross entropy because labels no longer integers after mixup
    optimizer1 = optim.SGD(net1.parameters(), lr=0.001, momentum=0.9) #as in tutorial
    optimizer2 = optim.SGD(net2.parameters(), lr=0.001, momentum=0.9) 
    optimizer3 = optim.Adam(net3.parameters(), lr=0.001)
    softmax_func = torch.nn.Softmax(dim=1) #used to get vector of probabilities

    Max_Epochs=10

    print("-" * 20 + "start" + "-" * 20)
    print("Original network for baseline")

    net1_time_elapsed = time.time() #measuring time taken
    for epoch in range(Max_Epochs):  # loop over the dataset multiple times
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            #one hot encoding for labels since we no longer use cross entropy loss but mse instead
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=10)
            inputs = inputs.to(device)
            labels = labels.to(device)
            one_hot_labels = 1.0*one_hot_labels.to(device) #1.0 to ensure to use right datatype
            
            # zero the parameter gradients
            optimizer1.zero_grad()
            # forward + backward + optimize
            outputs = net1(inputs)
            loss = criterion(outputs, one_hot_labels)
            loss.backward()
            optimizer1.step()
            del inputs, labels, one_hot_labels, data, outputs


        #after every epoch we report validation set performance
        print("Epoch: " + str(epoch+1))
        accuracy_net1_val=0.0
        cross_entropy_net1_val = 0.0
        final_loss_net1_val = 0.0
        for _, val_data in enumerate(validationloader, 0):
            val_images, val_labels = val_data
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            one_hot_val_labels = torch.nn.functional.one_hot(val_labels, num_classes=10)
            one_hot_val_labels = 1.0*one_hot_val_labels.to(device) #1.0 to ensure to use right datatype
            val_outputs = net1(val_images)
            val_softmax_output = softmax_func(val_outputs) #softmaxing so that we get probability vectors
            val_predictions = torch.argmax(val_outputs, 1, keepdim=False) #predicting by choosing indices of highest values across class dimension 
            final_loss_net1_val += criterion(val_outputs, one_hot_val_labels)
            accuracy_net1_val += torch.sum(val_predictions==val_labels)
            cross_entropy_net1_val += torch.sum(torch.log(val_softmax_output[torch.arange(val_softmax_output.shape[0]),val_labels]))
            del val_images, val_labels, val_data, val_outputs, val_softmax_output, val_predictions
        #metrics used are accuracy and cross-entropy
        final_loss_net1_val = final_loss_net1_val.item()
        accuracy_net1_val = 100.0*(accuracy_net1_val.item())/(validation_set_size) #test accuracy percentage
        cross_entropy_net1_val = - cross_entropy_net1_val.item()/(validation_set_size)
        print("Validation set MSE loss: " + str(final_loss_net1_val))
        print("Validation set accuracy: "+ str(accuracy_net1_val) + "%")
        print("Validation set cross entropy: " + str(cross_entropy_net1_val))

    
    net1_time_elapsed =  time.time() - net1_time_elapsed
    
    #loss and metrics on training for net 1
    final_loss_net1_train = 0.0
    accuracy_net1_train=0.0
    cross_entropy_net1_train = 0.0
    with torch.no_grad():
        for _, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=10)
            one_hot_labels = 1.0*one_hot_labels.to(device) #1.0 to ensure to use right datatype
            outputs = net1(inputs)
            final_loss_net1_train += criterion(outputs, one_hot_labels)
            del inputs, data, one_hot_labels
            train_softmax_output = softmax_func(outputs) #softmaxing so that we get probability vectors
            train_predictions = torch.argmax(outputs, 1, keepdim=False) #predicting by choosing indices of highest values across class dimension 
            accuracy_net1_train += torch.sum(train_predictions==labels)
            cross_entropy_net1_train += torch.sum(torch.log(train_softmax_output[torch.arange(train_softmax_output.shape[0]),labels]))
            del train_softmax_output, train_predictions, outputs, labels
        final_loss_net1_train = (final_loss_net1_train.item())*batch_size/train_set_size #average loss
        accuracy_net1_train = 100.0*(accuracy_net1_train.item())/(train_set_size) #train accuracy percentage
        cross_entropy_net1_train = - cross_entropy_net1_train.item()/(train_set_size) #train cross entropy



    #loss and metrics on testing for net 1
    final_loss_net1_test = 0.0
    accuracy_net1_test=0.0
    cross_entropy_net1_test = 0.0
    with torch.no_grad():
        for _, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=10)
            one_hot_labels = 1.0*one_hot_labels.to(device) #1.0 to ensure to use right datatype
            outputs = net1(inputs)
            final_loss_net1_test += criterion(outputs, one_hot_labels)
            del inputs, data, one_hot_labels
            test_softmax_output = softmax_func(outputs) #softmaxing so that we get probability vectors
            test_predictions = torch.argmax(outputs, 1, keepdim=False) #predicting by choosing indices of highest values across class dimension 
            accuracy_net1_test += torch.sum(test_predictions==labels)
            cross_entropy_net1_test += torch.sum(torch.log(test_softmax_output[torch.arange(test_softmax_output.shape[0]),labels]))
            del test_softmax_output, test_predictions, outputs, labels
        final_loss_net1_test = final_loss_net1_test.item() #test loss
        accuracy_net1_test = 100.0*(accuracy_net1_test.item())/(holdout_set_size) #test accuracy percentage
        cross_entropy_net1_test = - cross_entropy_net1_test.item()/(holdout_set_size) #test cross entropy

    #freeing memory
    del net1
    if (torch.cuda.is_available()):
        torch.cuda.empty_cache()
    print('Training done.')












    print("-" * 20 + "start" + "-" * 20)
    print("Mixup network with SGD")
    alpha = 0.4 #alpha mix parameter suggested to be kept between 0.1 and 0.4 in paper for sampling method 1
    sampling_method=1

    net2_time_elapsed = time.time()# measuring time taken
    for epoch in range(Max_Epochs):  # loop over the dataset multiple times
        for i, data in enumerate(trainloader, 0):
            Mixer = mixup(alpha, sampling_method) #mixup algorithm
            inputs, labels = data
            #one hot encoding for labels since we no longer use cross entropy loss but mse instead
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=10)
            inputs = inputs.to(device)
            one_hot_labels = 1.0*one_hot_labels.to(device) #1.0 to ensure to use right datatype
            #mixup inputs and labels
            mixed_inputs, mixed_one_hot_labels = Mixer.mix(inputs, one_hot_labels)

            # zero the parameter gradients
            optimizer2.zero_grad()
            # forward + backward + optimize
            outputs = net2(mixed_inputs)
            loss = criterion(outputs, mixed_one_hot_labels)
            loss.backward()
            optimizer2.step()
            del inputs, labels, data, one_hot_labels, mixed_inputs, mixed_one_hot_labels


        
        #after every epoch we report validation set performance
        print("Epoch: " + str(epoch+1))
        accuracy_net2_val=0.0
        cross_entropy_net2_val = 0.0
        final_loss_net2_val = 0.0
        for _, val_data in enumerate(validationloader, 0):
            val_images, val_labels = val_data
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            one_hot_val_labels = torch.nn.functional.one_hot(val_labels, num_classes=10)
            one_hot_val_labels = 1.0*one_hot_val_labels.to(device) #1.0 to ensure to use right datatype
            val_outputs = net2(val_images)
            val_softmax_output = softmax_func(val_outputs) #softmaxing so that we get probability vectors
            val_predictions = torch.argmax(val_outputs, 1, keepdim=False) #predicting by choosing indices of highest values across class dimension 
            final_loss_net2_val += criterion(val_outputs, one_hot_val_labels)
            accuracy_net2_val += torch.sum(val_predictions==val_labels)
            cross_entropy_net2_val += torch.sum(torch.log(val_softmax_output[torch.arange(val_softmax_output.shape[0]),val_labels]))
            del val_images, val_labels, val_data, val_outputs, val_softmax_output, val_predictions
        #metrics used are accuracy and cross-entropy
        final_loss_net2_val = final_loss_net2_val.item() #loss
        accuracy_net2_val = 100.0*(accuracy_net2_val.item())/(validation_set_size) #validation accuracy percentage
        cross_entropy_net2_val = - cross_entropy_net2_val.item()/(validation_set_size) #validation cross entropy
        print("Validation set MSE loss: " + str(final_loss_net2_val))
        print("Validation set accuracy: "+ str(accuracy_net2_val) + "%")
        print("Validation set cross entropy: " + str(cross_entropy_net2_val))
    net2_time_elapsed =  time.time() - net2_time_elapsed #time taken
    
    
    #loss and metrics on training for net 2
    final_loss_net2_train = 0.0
    accuracy_net2_train=0.0
    cross_entropy_net2_train = 0.0
    with torch.no_grad():
        for _, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=10)
            one_hot_labels = 1.0*one_hot_labels.to(device) #1.0 to ensure to use right datatype
            outputs = net2(inputs)
            final_loss_net2_train += criterion(outputs, one_hot_labels)
            del inputs, data, one_hot_labels
            train_softmax_output = softmax_func(outputs) #softmaxing so that we get probability vectors
            train_predictions = torch.argmax(outputs, 1, keepdim=False) #predicting by choosing indices of highest values across class dimension 
            accuracy_net2_train += torch.sum(train_predictions==labels)
            cross_entropy_net2_train += torch.sum(torch.log(train_softmax_output[torch.arange(train_softmax_output.shape[0]),labels]))
            del train_softmax_output, train_predictions, outputs, labels
        final_loss_net2_train = final_loss_net2_train.item()*batch_size/train_set_size
        accuracy_net2_train = 100.0*(accuracy_net2_train.item())/(train_set_size) #test accuracy percentage
        cross_entropy_net2_train = - cross_entropy_net2_train.item()/(train_set_size)
    
    #loss and metrics on testing for net 2
    final_loss_net2_test = 0.0
    accuracy_net2_test=0.0
    cross_entropy_net2_test = 0.0
    with torch.no_grad():
        for _, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=10)
            one_hot_labels = 1.0*one_hot_labels.to(device) #1.0 to ensure to use right datatype
            outputs = net2(inputs)
            final_loss_net2_test += criterion(outputs, one_hot_labels)
            del inputs, data, one_hot_labels
            test_softmax_output = softmax_func(outputs) #softmaxing so that we get probability vectors
            test_predictions = torch.argmax(outputs, 1, keepdim=False) #predicting by choosing indices of highest values across class dimension 
            accuracy_net2_test += torch.sum(test_predictions==labels)
            cross_entropy_net2_test += torch.sum(torch.log(test_softmax_output[torch.arange(test_softmax_output.shape[0]),labels]))
            del test_softmax_output, test_predictions, outputs, labels
        final_loss_net2_test = final_loss_net2_test.item()
        accuracy_net2_test = 100.0*(accuracy_net2_test.item())/(holdout_set_size) #test accuracy percentage
        cross_entropy_net2_test = - cross_entropy_net2_test.item()/(holdout_set_size)

    print('Training done.')

    # save trained model
    torch.save(net2.state_dict(), 'mixup_net_SGD.pt')
    del net2
    if (torch.cuda.is_available()):
        torch.cuda.empty_cache()
    print('Model saved.')

















    print("-" * 20 + "start" + "-" * 20)
    print("Mixup network with ADAM")
    alpha = 0.4 #alpha mix parameter suggested to be kept between 0.1 and 0.4 in paper for sampling method 1
    sampling_method=1

    net3_time_elapsed = time.time()
    for epoch in range(Max_Epochs):  # loop over the dataset multiple times
        final_loss_net3_train = 0.0 
        for i, data in enumerate(trainloader, 0):
            Mixer = mixup(alpha, sampling_method)
            inputs, labels = data
            #one hot encoding for labels since we no longer use cross entropy loss but mse instead
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=10)
            inputs = inputs.to(device)
            one_hot_labels = 1.0*one_hot_labels.to(device) #1.0 to ensure to use right datatype
            #mixup inputs and labels
            mixed_inputs, mixed_one_hot_labels = Mixer.mix(inputs, one_hot_labels)

            # zero the parameter gradients
            optimizer3.zero_grad()
            # forward + backward + optimize
            outputs = net3(mixed_inputs)
            loss = criterion(outputs, mixed_one_hot_labels)
            final_loss_net3_train +=loss
            loss.backward()
            optimizer3.step()
            del inputs, labels, data, one_hot_labels, mixed_inputs, mixed_one_hot_labels

        
        #after every epoch we report validation set performance
        print("Epoch: " + str(epoch+1))
        accuracy_net3_val=0.0
        cross_entropy_net3_val = 0.0
        final_loss_net3_val = 0.0
        for _, val_data in enumerate(validationloader, 0):
            val_images, val_labels = val_data
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            one_hot_val_labels = torch.nn.functional.one_hot(val_labels, num_classes=10)
            one_hot_val_labels = 1.0*one_hot_val_labels.to(device) #1.0 to ensure to use right datatype
            val_outputs = net3(val_images)
            val_softmax_output = softmax_func(val_outputs) #softmaxing so that we get probability vectors
            val_predictions = torch.argmax(val_outputs, 1, keepdim=False) #predicting by choosing indices of highest values across class dimension 
            final_loss_net3_val += criterion(val_outputs, one_hot_val_labels)
            accuracy_net3_val += torch.sum(val_predictions==val_labels)
            cross_entropy_net3_val += torch.sum(torch.log(val_softmax_output[torch.arange(val_softmax_output.shape[0]),val_labels]))
            del val_images, val_labels, val_data, val_outputs, val_softmax_output, val_predictions

        #metrics used are accuracy and cross-entropy
        final_loss_net3_val = final_loss_net3_val.item()
        accuracy_net3_val = 100.0*(accuracy_net3_val.item())/(validation_set_size) #test accuracy percentage
        cross_entropy_net3_val = - cross_entropy_net3_val.item()/(validation_set_size)
        print("Validation set MSE loss: " + str(final_loss_net3_val))
        print("Validation set accuracy: "+ str(accuracy_net3_val) + "%")
        print("Validation set cross entropy: " + str(cross_entropy_net3_val))
    net3_time_elapsed = time.time() - net3_time_elapsed #time taken
    
    #loss and metrics on training for net 3
    final_loss_net3_train = 0.0
    accuracy_net3_train=0.0
    cross_entropy_net3_train = 0.0
    with torch.no_grad():
        for _, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=10)
            one_hot_labels = 1.0*one_hot_labels.to(device) #1.0 to ensure to use right datatype
            outputs = net3(inputs)
            final_loss_net3_train += criterion(outputs, one_hot_labels)
            del inputs, data, one_hot_labels
            train_softmax_output = softmax_func(outputs) #softmaxing so that we get probability vectors
            train_predictions = torch.argmax(outputs, 1, keepdim=False) #predicting by choosing indices of highest values across class dimension 
            accuracy_net3_train += torch.sum(train_predictions==labels)
            cross_entropy_net3_train += torch.sum(torch.log(train_softmax_output[torch.arange(train_softmax_output.shape[0]),labels]))
            del train_softmax_output, train_predictions, outputs, labels
        final_loss_net3_train = final_loss_net3_train.item()*batch_size/train_set_size
        accuracy_net3_train = 100.0*(accuracy_net3_train.item())/(train_set_size) #test accuracy percentage
        cross_entropy_net3_train = - cross_entropy_net3_train.item()/(train_set_size) 




    #loss and metrics on testing for net 3
    final_loss_net3_test = 0.0
    accuracy_net3_test=0.0
    cross_entropy_net3_test = 0.0
    with torch.no_grad():
        for _, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=10)
            one_hot_labels = 1.0*one_hot_labels.to(device) #1.0 to ensure to use right datatype
            outputs = net3(inputs)
            final_loss_net3_test += criterion(outputs, one_hot_labels)
            del inputs, data, one_hot_labels
            test_softmax_output = softmax_func(outputs) #softmaxing so that we get probability vectors
            test_predictions = torch.argmax(outputs, 1, keepdim=False) #predicting by choosing indices of highest values across class dimension 
            accuracy_net3_test += torch.sum(test_predictions==labels)
            cross_entropy_net3_test += torch.sum(torch.log(test_softmax_output[torch.arange(test_softmax_output.shape[0]),labels]))
            del test_softmax_output, test_predictions, outputs, labels
        final_loss_net3_test = final_loss_net3_test.item()
        accuracy_net3_test = 100.0*(accuracy_net3_test.item())/(holdout_set_size) #test accuracy percentage
        cross_entropy_net3_test = - cross_entropy_net3_test.item()/(holdout_set_size)
    print('Training done.')

    # save trained model
    torch.save(net3.state_dict(), 'mixup_net_ADAM.pt')
    del net3
    if (torch.cuda.is_available()):
        torch.cuda.empty_cache()
    print('Model saved.')


    #Report a summary of loss values, speed, metric on training and validation. 
    print("summary of loss values, speed, metric on training and validation for the three networks")
    print("summary of loss values, speed, metric on training and validation for the three networks")
    print(f"{'' : <60}{'Original network' : <30}{'Mixup network with SGD' : <30}{'Mixup network with ADAM' : <30}")
    print(f"{'Final loss on training' : <60}{str(final_loss_net1_train) : <30}{str(final_loss_net2_train) : <30}{str(final_loss_net3_train) : <30}")
    print(f"{'Final loss on validation' : <60}{str(final_loss_net1_val) : <30}{str(final_loss_net2_val) : <30}{str(final_loss_net3_val) : <30}")
    print(f"{'Accuracy on training' : <60}{str(accuracy_net1_train) : <30}{str(accuracy_net2_train) : <30}{str(accuracy_net3_train) : <30}")
    print(f"{'Accuracy on validation' : <60}{str(accuracy_net1_val) : <30}{str(accuracy_net2_val) : <30}{str(accuracy_net3_val) : <30}")
    print(f"{'Cross entropy on training' : <60}{str(cross_entropy_net1_train) : <30}{str(cross_entropy_net2_train) : <30}{str(cross_entropy_net3_train) : <30}")
    print(f"{'Cross entropy on validation' : <60}{str(cross_entropy_net1_val) : <30}{str(cross_entropy_net2_val) : <30}{str(cross_entropy_net3_val) : <30}")
    print(f"{'Time for training and validation in seconds' : <60}{str(net1_time_elapsed) : <30}{str(net2_time_elapsed): <30}{str(net3_time_elapsed) : <30}")




    #Report a summary of loss values and the metrics on the holdout test set. Compare the results 
    #with those obtained during development.
    print("\nsummary of loss values,metric on testing for the three networks")
    print(f"{'' : <60}{'Original network(Resnet50)' : <30}{'Mixup network with SGD' : <30}{'Mixup network with ADAM' : <30}")
    print(f"{'Final loss on testing' : <60}{str(final_loss_net1_test) : <30}{str(final_loss_net2_test) : <30}{str(final_loss_net3_test) : <30}")
    print(f"{'Accuracy on testing' : <60}{str(accuracy_net1_test) : <30}{str(accuracy_net2_test) : <30}{str(accuracy_net3_test) : <30}")
    print(f"{'Cross entropy on testing' : <60}{str(cross_entropy_net1_test) : <30}{str(cross_entropy_net2_test) : <30}{str(cross_entropy_net3_test) : <30}")

    print("As we can see all 3 of final loss, accuracy and cross entropy are higher on the teting set than correponding values during validation.\nFor the original network and sgd, the values tend to be similar during testing and validation however for ADAM the difference is more significant. \nADAM had the longest training time however it has the worst performance. Furthermore, ADAM's performance seems to go up and down \nwhereas for the other two nets the performance seems more steady.")



if __name__ == '__main__':
    
    main()