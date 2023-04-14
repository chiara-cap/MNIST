import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Pre-processing operations
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# Create datasets for training & test
training_set = torchvision.datasets.MNIST('./data', train=True, transform=transform, download=True)
test_set = torchvision.datasets.MNIST('./data', train=False, transform=transform, download=True)

# Create validation set from training set
training_set, validation_set = train_test_split(training_set, test_size=0.2, random_state=25)

# Create data loaders for our datasets; shuffle for training and validation, not for test
training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False)

# Class labels
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# Report split sizes
print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_set)))
print('Test set has {} instances'.format(len(test_set)))

"""
# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


dataiter = iter(training_loader)
images, labels = next(dataiter)


# Create a grid from the images and show them
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)
plt.show()
print('  '.join(classes[labels[j]] for j in range(4)))
"""

# Defining the model
model = models.resnet18()
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model = model.cuda()

# Defining Loss Function
loss_fn = torch.nn.CrossEntropyLoss()

# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train_one_epoch(epoch_index):
    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(training_loader):

        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.0

            # Validation
            # We don't need gradients on to do reporting
            model.train(False)

            running_vloss = 0.0
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.cuda(), vlabels.cuda()
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(last_loss, avg_vloss))
            
        input = input.cpu()
        labels = labels.cpu()

    return last_loss


# Initializing in a separate cell, so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
epoch_number = 0

EPOCHS = 5

best_test_loss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number)

    # We don't need gradients on to do reporting
    model.train(False)

    running_test_loss = 0.0
    for i, test_data in enumerate(test_loader):
        test_inputs, test_labels = test_data
        test_inputs, test_labels = test_inputs.cuda(), test_labels.cuda()
        test_outputs = model(test_inputs)
        test_loss = loss_fn(test_outputs, test_labels)
        running_test_loss += test_loss

    avg_test_loss = running_test_loss / (i + 1)
    print('LOSS train {} test {}'.format(avg_loss, avg_test_loss))

    # Track best performance, and save the model's state
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

