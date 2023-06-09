import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image

import numpy as np
from resnet50_network_pt import MyResnet50 #resnet50
from mixup_class import mixup





def main():
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


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    

    #test set loading
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_set_size = len(testset) #size of testing set
    test_set_batch_size = 1000
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_set_batch_size, shuffle=False, num_workers=2, generator=generator)

    # Visualise your implementation, by saving to a PNG file “mixup.png”, a montage of 16 images 
    # with randomly augmented images that are about to be fed into network training.
    batch_size = 16
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, generator=generator)
    # example images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    
    alpha = 0.4 #alpha mix parameter suggested to be kept between 0.1 and 0.4 in paper
    sampling_method=1
    Mixer = mixup(alpha, sampling_method)

    images, _ = Mixer.mix(images, labels)
    

    im = Image.fromarray((torch.cat(images.split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8'))
    im.save("mixup.png")
    del trainloader, batch_size, dataiter, images, im




    #Train a new ResNet classification network with mixup data augmentation, for each of the two 
    #sampling methods, with 10 epochs.

    #training
    batch_size = 20
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, generator=generator)
    net = MyResnet50()
    net.to(device) #making sure we're on right device

    # loss and optimiser
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss() #switched to MSE instead of cross entropy because labels no longer integers after mixup
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #as in tutorial

    alpha = 0.4 #alpha mix parameter suggested to be kept between 0.1 and 0.4 in paper for sampling method 1
    Max_Epochs=10

    #sampling method 1
    sampling_method=1
    print("-" * 20 + "start" + "-" * 20)
    print("Sampling method 1 (alpha value: " + str(alpha) +")\nwith pre-trained Resnet initialisation")
    for epoch in range(Max_Epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            Mixer = mixup(alpha, sampling_method)
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            #one hot encoding for labels since we no longer use cross entropy loss but mse instead
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=10)

            inputs = inputs.to(device)
            one_hot_labels = 1.0*one_hot_labels.to(device) #1.0 to ensure to use right datatype
            
            #mixup inputs and labels
            mixed_inputs, mixed_one_hot_labels = Mixer.mix(inputs, one_hot_labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(mixed_inputs)
            #print(torch.argmax(outputs, 1, keepdim=True))

            loss = criterion(outputs, mixed_one_hot_labels)
            
            loss.backward()
            optimizer.step()

            # # print statistics
            # running_loss += loss.item()
            # if i % 2000 == 1999:    # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #         (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0
        
        #after every epoch we report test set performance
        print("Epoch: " + str(epoch+1))
        accuracy=0.0
        for _, test_data in enumerate(testloader, 0):
            test_images, test_labels = test_data
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)

            test_outputs = net(test_images)
            test_predictions = torch.argmax(test_outputs, 1, keepdim=False) #predicting by choosing indices of highest values across class dimension 
            accuracy += torch.sum(test_predictions==test_labels)

        accuracy = 100.0*(accuracy.item())/(test_set_size) #test accuracy percentage
        print("Testing accuracy: "+ str(accuracy) + "%")
        

    print('Training done.')

    # save trained model
    torch.save(net.state_dict(), 'sampling_method_one_model.pt')
    print('Model saved.')




    #sampling method 2
    sampling_method=2
    print("-" * 20 + "start" + "-" * 20)
    print("Sampling method 2 with pre-trained Resnet initialisation.\nLamda values sampled uniformly from [0,0.5)")


    net2 = MyResnet50()
    net2.to(device) #making sure we're on right device
    optimizer = optim.SGD(net2.parameters(), lr=0.001, momentum=0.9) #as in tutorial

    for epoch in range(Max_Epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            Mixer = mixup(alpha, sampling_method)
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            #one hot encoding for labels since we no longer use cross entropy loss but mse instead
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=10)

            inputs = inputs.to(device)
            one_hot_labels = 1.0*one_hot_labels.to(device) #1.0 to ensure to use right datatype
            
            #mixup inputs and labels
            mixed_inputs, mixed_one_hot_labels = Mixer.mix(inputs, one_hot_labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net2(mixed_inputs)
            #print(torch.argmax(outputs, 1, keepdim=True))

            loss = criterion(outputs, mixed_one_hot_labels)
            
            loss.backward()
            optimizer.step()

            # # print statistics
            # running_loss += loss.item()
            # if i % 2000 == 1999:    # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #         (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0
        
        #after every epoch we report test set performance
        print("Epoch: " + str(epoch+1))
        accuracy=0.0
        for _, test_data in enumerate(testloader, 0):
            test_images, test_labels = test_data
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)

            test_outputs = net2(test_images)
            test_predictions = torch.argmax(test_outputs, 1, keepdim=False) #predicting by choosing indices of highest values across class dimension 
            accuracy += torch.sum(test_predictions==test_labels)

        accuracy = 100.0*(accuracy.item())/(test_set_size) #test accuracy percentage
        print("Testing accuracy: "+ str(accuracy) + "%")
        

    print('Training done.')

    # save trained model
    torch.save(net2.state_dict(), 'sampling_method_two_model.pt')
    print('Model saved.')

    #Visualise your results, by saving to a PNG file “result.png”, a montage of 36 test images with 
    #printed messages clearly indicating the ground-truth and the predicted classes for each. 
    batch_size = 36
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2, generator=generator)
    dataiter = iter(testloader)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # inference
    images, labels = next(dataiter)
    # save to images
    im = Image.fromarray((torch.cat(images.split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8'))
    im.save("result.png")

    images = images.to(device)
    labels = labels.to(device)
    print('Ground-truth:\n', ' '.join('%5s' % classes[labels[j]] for j in range(36)))

    outputs = net(images)
    predicted = torch.argmax(outputs, 1, keepdim=False)
    print('Predicted by network trained with sampling method 1:\n', ' '.join('%5s' % classes[predicted[j]] for j in range(36)))

    outputs = net2(images)
    predicted = torch.argmax(outputs, 1, keepdim=False)
    print('Predicted by network trained with sampling method 2:\n', ' '.join('%5s' % classes[predicted[j]] for j in range(36)))

    

if __name__ == '__main__':
    main()