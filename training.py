import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

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

# Report split sizes
print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_set)))
print('Test set has {} instances'.format(len(test_set)))
