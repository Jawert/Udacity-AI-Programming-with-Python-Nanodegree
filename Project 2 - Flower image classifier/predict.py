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
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--json_file', type = str, default = 'cat_to_name.json', help = 'Specify the json file to be used.')
parser.add_argument('--test_file', type = str, default = 'flowers/valid/13/image_05759.jpg', help = 'Specify the test image to use.')
parser.add_argument('--checkpoint_file', type = str, default = 'checkpoint.pth', help = 'Specify the checkpoint file to load from.')
parser.add_argument('--topk', type = int, default = 5, help = 'Specify the number of "top k" predictions to use.')
parser.add_argument('--gpu', default = 'gpu', type = str, help = 'Specify if you want to use the GPU or CPU.')

# Maps the parser arguments to variables for easier access later
p_inputs = parser.parse_args()

json_file = p_inputs.json_file
test_file = p_inputs.test_file
checkpoint_file = p_inputs.checkpoint_file
topk = p_inputs.topk
gpu = p_inputs.gpu

# Import the specified json file to read the names of the images.
with open(json_file, 'r') as f:
    cat_to_name = json.load(f)
                    
# Load the model from the saved checkpoint file.
def load_checkpoint(checkpoint_file = checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model = checkpoint['architecture']
    learning_rate = checkpoint['learning_rate']
    hidden_layers = checkpoint['hidden_layers']
    gpu = checkpoint['gpu']
    epochs = checkpoint['epochs']
    dropout = checkpoint['dropout']
    classifier = checkpoint['classifier']
    state_dict = checkpoint['state_dict']
    class_to_idx = checkpoint['class_to_idx']

    if model == 'vgg16':
        model = models.vgg16(pretrained = True)
    elif model == 'densenet121':
        model = models.densenet121(pretrained = True)
    elif model == 'alexnet':
        model = models.alexnet(pretrained = True)

    # Set the model to match the parameters determined in the classiier, class_to_idx, and state_dict.
    model.classifier = classifier
    model.class_to_idx = class_to_idx
    model.load_state_dict(state_dict)
    
    # Ensure that gradient descent is turned off.
    for param in model.parameters():
        param.requires_grad = False

    return model                 

# Load the model from the checkpoint
model = load_checkpoint()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    pil_image = Image.open(image).convert('RGB')
    
    pil_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    pil_image = pil_transforms(pil_image)
    
    return pil_image

# Run test image through the 'process_image' code.
image_check = process_image(test_file)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk = 5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # DONE: Implement the code to predict the class from an image file
    test_image = process_image(image_path)
    test_image = test_image.float().unsqueeze_(0)
    
    # Force model onto CPU because the GPU was causing issues.
    model.to('cpu')
    
    with torch.no_grad():
        output = model.forward(test_image)
    
    prediction = F.softmax(output.data, dim = 1)
    probabilities, indices = prediction.topk(topk)
    probabilities = probabilities.cpu().numpy()[0]
    indices = indices.cpu().numpy()[0]
    
    idx_to_class = {v:k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[x] for x in indices]
    max_prob = probabilities.max()
    label = [cat_to_name[x] for x in classes]
    
    return probabilities, classes, label, max_prob

# Calculate the probabilities and classes
probabilities, classes, label, max_prob = predict(test_file, model)


# Check that the probabilities look realistic.
print("The most likely classes are:")
print(classes)
print("\n")
print("The associated probabilities with these clases are:")
print(probabilities)
print("\n")
print("The labels associated with these classes are:")
print([cat_to_name[x] for x in classes])
print("\n")
print(f'The image used is: {p_inputs.test_file}')
print(f'The predicted class is: {classes[0]}')
print(f'The probability of being correct is: {max_prob*100:.2f}%')
