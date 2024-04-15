import torch
import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

# Check whether Nvidia GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():  # Multi-Process Service
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"{device} device is available")

# Transformations
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])

# Hyperparameters
labels_size = 6
epochs_size = 11
batch_size = 100
learning_rate = 0.001

# Paths
train_path = 'input/intel-image-classification/seg_train/seg_train/'
test_path = 'input/intel-image-classification/seg_test/seg_test/'

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


# CNN model
# kernel = filter
# padding = 0
# stride = 1
# input_channel = 3
# output_channel = 32, 64, 128
# output_size = (input_size - kernel_size + 2 * padding) / stride + 1
# output after first convolution = (150 - 3 + 0) / 1 + 1 = 148

# depth after first convolution = 32
# height and width after first convolution = 148
# filter height and width after first convolution = 3

# output after first pooling = 148 / 2 = 74

# depth after first pooling = 32
# height and width after first pooling = 74
# filter height and width after first pooling = 3


# output after second convolution = (74 - 3 + 0) / 1 + 1 = 72

# depth after second convolution = 64
# height and width after second convolution = 72
# filter height and width after second convolution = 3


# output after second pooling = 72 / 2 = 36

# depth after second pooling = 64
# height and width after second pooling = 36
# filter height and width after second pooling = 3


# output after third convolution = (36 - 3 + 0) / 1 + 1 = 34

# depth after third convolution = 128
# height and width after third convolution = 34
# filter height and width after third convolution = 3

# output after third pooling = 34 / 2 = 17
# output after flattening = 128 * 17 * 17
# output after first fully connected layer = 512
# output after second fully connected layer = 6

# depth of the first layer is 3 because the input image has 3 channels (RGB)
# filter width and height are 3
# filter depth is 32


class CNN(nn.Module):
    def __init__(self, labels_size):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.fc1 = nn.Linear(in_features=128 * 17 * 17, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=labels_size)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 128 * 17 * 17)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = CNN(labels_size=6).to(device)
summary(model, input_size=(100, 3, 150, 150))
# Pytorch summary seems to change the device of the model
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_count = len(glob.glob(train_path + '/**/*.jpg'))
test_count = len(glob.glob(test_path + '/**/*.jpg'))
print(train_count, test_count)

# Train and validate the model along with plotting the loss and accuracy
# We need to plot the loss and accuracy of the training and validation data
# over the number of epochs to see how well the model is learning

# (learning curves) of the training and validation error over "time"
# (have number of epochs on the x-axis, and both errors on the y-axis,

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

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
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

    print(f"Epoch {epoch + 1}/{epochs_size} => "
          f"Train Loss: {train_loss[-1]:.4f}, Train Accuracy: {train_accuracy[-1]:.2f}%, "
          f"Validation Loss: {val_loss[-1]:.4f}, Validation Accuracy: {val_accuracy[-1]:.2f}%")

# # Plotting
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(10, 7))
# plt.plot(train_loss, label='Train Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# plt.figure(figsize=(10, 7))
# plt.plot(train_accuracy, label='Train Accuracy')
# plt.plot(val_accuracy, label='Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

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
