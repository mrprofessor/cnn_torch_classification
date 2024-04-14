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

# Paths
train_path = 'input/intel-image-classification/seg_train/seg_train/'
test_path = 'input/intel-image-classification/seg_test/seg_test/'

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
# output after first pooling = 148 / 2 = 74
# output after second convolution = (74 - 3 + 0) / 1 + 1 = 72
# output after second pooling = 72 / 2 = 36
# output after third convolution = (36 - 3 + 0) / 1 + 1 = 34
# output after third pooling = 34 / 2 = 17
# output after flattening = 128 * 17 * 17
# output after first fully connected layer = 512
# output after second fully connected layer = 6


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

# Training
for epoch in range(epochs_size):
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        images, labels = data

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_loss += loss.cpu().data * images.size(0)
        _, prediction = torch.max(outputs.data, 1)

        train_accuracy += int(torch.sum(prediction == labels.data))

        if i % 100 == 99:
            print(i)
            print(f'Epoch: {epoch + 1}, Batch: {batch_size}, Loss: {running_loss / 100}')
            running_loss = 0.0

    train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count

    print(f'Epoch: {epoch + 1}, Loss: {train_loss}, Accuracy: {train_accuracy}')

    # Validation
    model.eval()
    val_accuracy = 0.0
    for i, data in enumerate(val_loader):

        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        val_accuracy += int(torch.sum(prediction == labels.data))

    val_accuracy = val_accuracy / val_size
    print(f'Epoch: {epoch + 1}, Validation Accuracy: {val_accuracy}')

print('Finished Training')
# PATH = './cnn.pth'
# torch.save(model.state_dict(), PATH)

print('Evaluate the model accuracy with test data')
# Evaluate the model accuracy with test data
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (prediction == labels).sum().item()

    print(f'Accuracy: {100 * correct / total}')
