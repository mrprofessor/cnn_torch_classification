import torch
import glob
import torchvision
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

# Load data
train_data = ImageFolder(train_path, transform=transform)
test_data = ImageFolder(test_path, transform=transform)

# Split data
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = random_split(train_data, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 18 * 18, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, input_data):
        output = self.pool(nn.functional.relu(self.conv1(input_data)))
        output = self.pool(nn.functional.relu(self.conv2(output)))
        output = self.pool(nn.functional.relu(self.conv3(output)))
        output = output.view(-1, 128 * 18 * 18)
        output = nn.functional.relu(self.fc1(output))
        output = self.fc2(output)
        return output


model = CNN(num_classes=6).to(device)
summary(model, input_size=(64, 3, 150, 150))
model = model.to(device) # Pytorch summary seems to change the device of the model

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_count = len(glob.glob(train_path+'/**/*.jpg'))
test_count = len(glob.glob(test_path+'/**/*.jpg'))
print(train_count, test_count)

# Training
num_of_epochs = 10
for epoch in range(num_of_epochs):
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        images, labels = data

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_loss += loss.cpu().data * images.size(0)
        _, prediction = torch.max(outputs.data, 1)

        train_accuracy += int(torch.sum(prediction == labels.data))

        if i % 100 == 99:
            print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 100}')
            running_loss = 0.0

    train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count

    print(f'Epoch: {epoch + 1}, Training Accuracy: {train_accuracy}, Training Loss: {train_loss}')

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

    if val_accuracy > 0.9:
        torch.save(model.state_dict(), 'best_model.pth')
        break

    torch.save(model.state_dict(), 'best_model.pth')
