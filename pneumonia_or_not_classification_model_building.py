#Imaport libraries
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Hyper parameters
num_epochs = 10
num_classes = 2
batch_size = 32
learning_rate = 0.0001

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#Specify transforms using torchvision library

transformation = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# load each dataset and apply transformations
train_set = datasets.ImageFolder(
    "/home/mahesh/machine-learning/kaggle/chest-xray-pneumonia/train", transform=transformation)

valid_set = datasets.ImageFolder(
    "/home/mahesh/machine-learning/kaggle/chest-xray-pneumonia/test", transform=transformation)

#Put into Dataloader 

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True)

valid_loader = torch.utils.data.DataLoader(
    valid_set, batch_size=batch_size, shuffle=True)

print("Training data size: " + str(len(train_loader.dataset)))
print("Test data size: " + str(len(valid_loader.dataset)))

class_to_idx = train_set.class_to_idx
print(class_to_idx)

model = models.vgg16(pretrained=True)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, num_classes)

model = model.cuda()


#set error function using torch.nn as nn lib
criterion = nn.CrossEntropyLoss()
#set the optimiser function using torch.optim as optim library
optimizer = optim.Adam(model.parameters(), lr= learning_rate)


#Training
min_loss = 1000
pre_best_min_loss = 10000
best_accuracy = 0.0
accuracy = 0.0
total_step = len(train_loader)
for epoch in range(num_epochs):
    print("Starting epoch :" + str(epoch))
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        #forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #Backward and optimise
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < min_loss:
            min_loss = loss.item()
        print('Step [{}/{}], Loss: {:.4f}'
            .format(i+1, total_step, min_loss))
        #break
    #save model with lowest loss
    if min_loss < pre_best_min_loss:
        pre_best_min_loss = min_loss
    
    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
          .format(epoch+1, num_epochs, i+1, total_step, min_loss))

    #Test the model
    #model.train = False
    model.eval()
    val_correct  = 0
    val_total = 0
    optimizer.zero_grad()
    for i, (images, labels) in enumerate(valid_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        val_total += labels.size(0)
        val_correct += (predicted == labels).sum().item()
    accuracy = 100*(val_correct/ val_total)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        print("saving best model with best accuracy " + str(best_accuracy))
        torch.save(model, 'pneumonia_or_not_classification_vgg16_model_v1.pt')
#print('Test Accuracy of the model on the 10000 test images: {} %'.format(best_accuracy))
