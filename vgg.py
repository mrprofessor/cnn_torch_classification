#%%
import torch
import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights, vgg19, VGG19_Weights
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import matplotlib.pyplot as plt
import numpy as np
#%%
# Check whether Nvidia GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():  # Multi-Process Service
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"{device} device is available")
#%%
# Hyperparameters
image_width = 150
image_height = 150
epochs_size = 15
batch_size = 128
dropout_rate = 0.5
learning_rate = 0.0001
gamma = 0.088

train_transforms = transforms.Compose([
    transforms.Resize(size=(150 , 150)) ,
    transforms.ColorJitter(0.4,0.5,0.5,0.2),
    transforms.RandomHorizontalFlip(p=0.5) ,
    transforms.RandomCrop(size=(150,150)),
    transforms.ToTensor(),
    transforms.Normalize((0.425, 0.415, 0.405), (0.205, 0.205, 0.205))
])


test_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize((0.425, 0.415, 0.405), (0.255, 0.245, 0.235))
])

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

print(f"Epochs: {epochs_size}, " +
        f"Batch Size: {batch_size}, " +
        f"Learning Rate: {learning_rate}, " +
        f"Dropout Rate: {dropout_rate}, " +
        f"Gamma: {gamma}")


print(f"Epochs: {epochs_size}, Batch Size: {batch_size}, Learning Rate: {learning_rate}, Dropout Rate: {dropout_rate}")

# Classes
classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
class_size = len(classes)
#%%
# Paths
train_path = '../input/intel-image-classification/seg_train/seg_train/'
test_path = '../input/intel-image-classification/seg_test/seg_test/'

# Load data
train_data = ImageFolder(train_path, transform=transform)
test_data = ImageFolder(test_path, transform=transform)

# Split data
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = random_split(train_data, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
#%%
model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.required_grad = False

cnn_summary = summary(model, input_size=(1, 3, image_width, image_height))
print(cnn_summary)
# Pytorch summary seems to change the device of the model
model = model.to(device)
#%%


# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
schedule_learning = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer , milestones=[3 , 6 ] ,
                                                        gamma=gamma)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

train_count = len(glob.glob(train_path + '/**/*.jpg'))
test_count = len(glob.glob(test_path + '/**/*.jpg'))
print(train_count, test_count)
#%%
# Train and validate the model
train_loss = []
val_loss = []
train_accuracy = []
val_accuracy = []

for epoch in range(epochs_size):
    model.train()
    train_loss_current = 0.0
    correct_train = 0
    total_train = 0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_current += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss.append(train_loss_current / len(train_loader))
    train_accuracy.append(100 * correct_train / total_train)

    model.eval()
    val_loss_current = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss_current += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss.append(val_loss_current / len(val_loader))
    val_accuracy.append(100 * correct_val / total_val)

    schedule_learning.step()

    print(f"Epoch {epoch + 1}/{epochs_size} => "
          f"Train Loss: {train_loss[-1]:.4f}, Train Accuracy: {train_accuracy[-1]:.2f}%, "
          f"Validation Loss: {val_loss[-1]:.4f}, Validation Accuracy: {val_accuracy[-1]:.2f}%")

# print val_loss and train_loss
print(f"train_loss = {train_loss}")
print(f"val_loss = {val_loss}")
print(f"train_accuracy = {train_accuracy}")
print(f"val_accuracy = {val_accuracy}")
#%%
# Plot training and validation loss and accuracy horizontally
train_loss = [0.4089617403647439, 0.14766680870459162, 0.0878399213860658, 0.05181924837630835, 0.04469992128856988, 0.03703689131229608, 0.03269399731388231, 0.03140917623981791, 0.030625294688930313, 0.028464247299284165, 0.030446153053410606, 0.02754367107858839, 0.028849932144194925, 0.026326349775031718, 0.027071791887547905]
val_loss = [0.21030825325711208, 0.1884681289507584, 0.20806442277336662, 0.19605287371880628, 0.20163738253441724, 0.1852291917259043, 0.1958043502474373, 0.20947837198830463, 0.1888507976281372, 0.1914344159886241, 0.2111612685363401, 0.19679948617704213, 0.2051759207282554, 0.19759512493725528, 0.19901434526863424]
train_accuracy = [87.1559633027523, 95.00311748463525, 97.01612184911374, 98.34327959383629, 98.64612095840384, 98.86879843235059, 99.10038300525518, 99.10929010421306, 99.08256880733946, 99.13601140108666, 99.08256880733946, 99.21617529170749, 99.11819720317092, 99.19836109379175, 99.09147590629732]
val_accuracy = [92.55432846455291, 93.69433558959743, 93.55183469896687, 94.15746348414677, 93.836836480228, 94.33558959743499, 94.29996437477735, 93.6587103669398, 94.22871392946206, 94.15746348414677, 93.80121125757036, 94.22871392946206, 93.58745992162451, 94.37121482009263, 94.37121482009263]


plt.figure(figsize=(20, 7))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#%%
# Finally Test the model
model.eval()
correct_test = 0
total_test = 0
with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct_test / total_test:.2f}%")
#%%
# Identify the worst classified example of each of the six classes. "Worst" is defined as having the max difference between probability of predicted but wrong class and probability of correct class.
model.eval()
worst_classified = {
    'buildings': { 'difference': 0.0, 'image': None},
    'forest': {'difference': 0.0, 'image': None},
    'glacier': { 'difference': 0.0, 'image': None},
    'mountain': {'difference': 0.0, 'image': None},
    'sea': {'difference': 0.0, 'image': None},
    'street': {'difference': 0.0, 'image': None},
}

with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        img = inputs[i].view(1,3,150,150)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        for j in range(len(predicted)):
            if predicted[j] != labels[j]:
                difference = outputs[j][predicted[j]] - outputs[j][labels[j]]
                if difference > worst_classified[classes[labels[j]]]['difference']:
                    worst_classified[classes[labels[j]]]['difference'] = difference
                    worst_classified[classes[labels[j]]]['image'] = inputs[j].cpu().numpy().transpose((1, 2, 0))



# plot all the images horizontally and show both actual and predicted classes

# getting error in this code
# Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
plt.figure(figsize=(20, 30))
i = 0
for key, value in worst_classified.items():
    plt.subplot(2, 3, i + 1)
    plt.title(f"Predicted: {key}, Actual: {classes[(classes.index(key) + 1) % 6]}, Difference: {value['difference']:.4f}")
    plt.axis('off')
    plt.imshow(value['image'])
    i = i + 1
plt.savefig('chart.png')
#%%
