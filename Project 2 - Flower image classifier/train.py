'''
@Author: Jacob Wert
@Title: Image Classifier Training File
'''

# Import all of the necessary packages and modules.
import json
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms, models, utils
import matplotlib.pyplot as plt
from collections import OrderedDict
import argparse
  
parser = argparse.ArgumentParser()

parser.add_argument('--arch', type = str, default = 'vgg16', help = 'Specify the pre-trained model architecture to use. By default this model uses VGG16')
parser.add_argument('--learning_rate', type=float, default = 0.001, help = 'Specify the learning rate of the model - default is 0.001')
parser.add_argument('--hidden_layers', type = int, default = 1024, help='Specify the number of hidden layers in the model.')
parser.add_argument('--gpu', default = 'gpu', type = str, help='Specify if you want the model to run on the GPU or CPU. Default is the GPU.')
parser.add_argument('--epochs', type = int, default = 5, help='Specify the number of cycles you want the model to run for training. Default is 5.')
parser.add_argument('--dropout', type = float, default = 0.5, help='Specify probability rate for dropouts. Default is 50% (0.5).')
parser.add_argument('--save_dir', type = str, default = './checkpoint.pth', help = 'Specify the save directory for a checkpoint of the model. Default is ./checkpoint.pth')

# Maps the parser arguments to variables for easier access later
p_inputs = parser.parse_args()

architecture = p_inputs.arch
learning_rate = p_inputs.learning_rate
hidden_layers = p_inputs.hidden_layers
gpu = p_inputs.gpu
epochs = p_inputs.epochs
dropout = p_inputs.dropout
save_dir = p_inputs.save_dir

# Pull in data
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define transformations for the training dataset.

data_transforms = {
    "training": transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(20),
                                    transforms.RandomResizedCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])]),
    
    "validation": transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])]),
    
    "testing": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])
}

# DONE: Load the datasets with ImageFolder
image_datasets = {
    "training": datasets.ImageFolder(train_dir, transform = data_transforms["training"]),
    "validation": datasets.ImageFolder(valid_dir, transform = data_transforms["validation"]),
    "testing": datasets.ImageFolder(test_dir, transform = data_transforms["testing"])
}
# DONE: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {
    "training": torch.utils.data.DataLoader(image_datasets["training"], batch_size = 64, shuffle = True),
    "validation": torch.utils.data.DataLoader(image_datasets["validation"], batch_size = 32),
    "testing": torch.utils.data.DataLoader(image_datasets["testing"], batch_size = 20)
}

# Make sure that the user is only using a supported model.
if not p_inputs.arch.startswith("vgg16") and not p_inputs.arch.startswith("densenet121") and not p_inputs.arch.startswith("alexnet"):
    print("This image classifier only supports VGG16, DenseNet121, and Alexnet. Please one of these models and try again.")
    exit(1)

else:
    print(f"You have selected the {p_inputs.arch} network.")

# Defining the model based on inputs
def Classifier(architechture = 'vgg16', dropout = 0.5, hidden_layers = 1024):
    if architecture == 'vgg16':
        model = models.vgg16(pretrained = True)
        input_size = 25088
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained = True)
        input_size = 1024
    elif architecture == 'alexnet':
        model = models.alexnet(pretrained = True)
        input_size = 9216

    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(dropout)),
            ('fc1', nn.Linear(input_size, hidden_layers)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(hidden_layers, 256)),
            ('output', nn.Linear(256, 102)),
            ('softmax', nn.LogSoftmax(dim = 1))]))
    
    model.classifier = classifier
    
    return model

# Set the model to match the classier that was specified.
model = Classifier(architecture, dropout, hidden_layers)

# The criterion must be negative log likehood loss because we are using Softmax
criterion = nn.NLLLoss()

# Set the optimizer to use "Adam", and the learning rate to the user specified rate.
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

def train_model(model = model, criterion = criterion, optimizer = optimizer, epochs = epochs, gpu = gpu):
    '''
    This function trains the model based on the user specified information.
    @model: The model that was specified by the user, and modified by Classifier().
    @criterion: Criterion set to Negative Log Likelihood Loss due to using the Softmax function.
    @optimizer: Sets the optimizer to the 'Adam' model.
    @epochs: The number of epochs (training cycles) as specified by the user.
    @gpu: Whether or not to train on the GPU.
    '''
    steps = 0
    print_every = 30
    
    print("We will now begin training the model...\n")
    
    # If the user opted to use the GPU, set the model to GPU.
    if gpu == 'gpu':
        model.to('cuda')
    
    # For loop to train the model based on the user specified number of epochs
    for e in range(epochs):
        running_loss = 0

        for ii, (images, labels) in enumerate(dataloaders["training"]):
            steps += 1

            # Moving images and labels to the GPU if specified to do so by the user.
            if gpu == 'gpu':
                images, labels = images.to('cuda'), labels.to('cuda')

            # Turning off gradient descent.
            optimizer.zero_grad()

            # Calculate the loss and backpropogate the weights (adjustments).
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Adjusting running loss
            running_loss += loss.item()

            # Calculate the validation loss and accuracy every 20 steps (arbitrary number, seems to work fine).
            if steps % print_every == 0:
                # Make sure that the model is on evaluation mode when checking the validation loss and accuracy.
                model.eval()

                val_loss = 0
                accuracy = 0

                # Calculating validation loss & accuracy.
                for ii, (val_images, val_labels) in enumerate(dataloaders["validation"]):

                    # Double check that the device is set to the GPU if specified by the user.
                    if gpu == 'gpu':
                        val_images, val_labels = val_images.to('cuda'), val_labels.to('cuda')

                    # Turning off gradient descent for training purposes.
                    with torch.no_grad():
                        val_outputs = model.forward(val_images)
                        val_loss = criterion(val_outputs, val_labels)

                        # Calculate the probability based on the validation outputs.
                        ps = torch.exp(val_outputs)
                        top_p, top_class = ps.topk(1, dim = 1)
                        equals = top_class == val_labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                # Print out the specified metrics as we go to know what is happening with our model.
                print('Epoch: {} / {}..'.format(e + 1, epochs),
                      'Training Loss: {:.3f}..'.format(running_loss / print_every),
                      'Validation Loss: {:.3f}..'.format(val_loss / len(dataloaders["validation"])),
                      'Validation Accuracy: {:.3f}..'.format(accuracy / len(dataloaders["validation"])))

                running_loss = 0
                
                # Make sure to set the model back into 'train' mode when done.
                model.train()

    print("Training complete!")

# Run the function to train the model based on the specified model, criterion, optimizer, epochs, and gpu setting.
train_model(model, criterion, optimizer, epochs, gpu)

def test_checker(gpu = gpu):
    num_correct = 0
    total = 0
    
    if gpu == 'gpu':
        model.to('cuda')
    
    with torch.no_grad():
        for ii, (images, labels) in enumerate(dataloaders['testing']):
            if gpu == 'gpu':
                images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            num_correct += (predicted == labels).sum().item()
        print('The network performed with {:.4f}% accuracy.'.format((num_correct / total) * 100))

test_checker()

# Saving a checkpoint for use later.
model.class_to_idx = image_datasets["training"].class_to_idx

checkpoint = {'architecture': model,
              'learning_rate': learning_rate,
              'hidden_layers': hidden_layers,
              'gpu': gpu,
              'epochs': epochs,
              'dropout': dropout,
              'classifier': model.classifier,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}

torch.save(checkpoint, save_dir)