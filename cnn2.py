# %%
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

# %%
# Hyperparameters
image_width = 150
image_height = 150
epochs_size = 15
batch_size = 64
dropout_rate = 0.5
learning_rate = 0.001
gamma = 0.055

train_transforms = transforms.Compose([
    transforms.Resize(size=(150, 150)),
    transforms.ColorJitter(0.4, 0.5, 0.5, 0.2),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(size=(150, 150)),
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
# %%
# Check whether Nvidia GPU is available
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():  # Multi-Process Service
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"{device} device is available")
# %%
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


# %%
class ImprovedCNN(nn.Module):
    def __init__(self, class_size, dropout_rate):
        super(ImprovedCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # size = 150x150x32
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # size = 75x75x64
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # size = 37x37x128
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # size = 18x18x256
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        # size = 9x9x512
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # size = 4x4x512
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(in_features=512 * 4 * 4, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=class_size)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)
        self.batch_norm5 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.batch_norm1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(nn.functional.relu(self.batch_norm2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(nn.functional.relu(self.batch_norm3(self.conv3(x))))
        x = self.dropout(x)
        x = self.pool(nn.functional.relu(self.batch_norm4(self.conv4(x))))
        x = self.dropout(x)
        x = self.pool(nn.functional.relu(self.batch_norm5(self.conv5(x))))
        x = self.dropout(x)
        x = x.view(-1, 512 * 4 * 4)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = ImprovedCNN(class_size=6, dropout_rate=dropout_rate).to(device)
cnn_summary = summary(model, input_size=(1, 3, image_width, image_height))
print(cnn_summary)
# Pytorch summary seems to change the device of the model
model = model.to(device)

# %%
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#schedule_learning = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[3, 6],gamma=gamma)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

train_count = len(glob.glob(train_path + '/**/*.jpg'))
test_count = len(glob.glob(test_path + '/**/*.jpg'))
print(train_count, test_count)
# %%
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

    # schedule_learning.step()

    print(f"Epoch {epoch + 1}/{epochs_size} => "
          f"Train Loss: {train_loss[-1]:.4f}, Train Accuracy: {train_accuracy[-1]:.2f}%, "
          f"Validation Loss: {val_loss[-1]:.4f}, Validation Accuracy: {val_accuracy[-1]:.2f}%")

# print val_loss and train_loss
print(f"train_loss = {train_loss}")
print(f"val_loss = {val_loss}")
print(f"train_accuracy = {train_accuracy}")
print(f"val_accuracy = {val_accuracy}")
# %%


plt.figure(figsize=(20, 7))
plt.subplot(1, 2, 1)
plt.plot(train_loss[:15], label='Train Loss')
plt.plot(val_loss[:15], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracy[:15], label='Train Accuracy')
plt.plot(val_accuracy[:15], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# %%
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
# %%
# Identify the worst classified example of each of the six classes. "Worst" is defined as having the max difference between probability of predicted but wrong class and probability of correct class.
model.eval()
worst_classified = {
    'buildings': {'difference': 0.0, 'image': None},
    'forest': {'difference': 0.0, 'image': None},
    'glacier': {'difference': 0.0, 'image': None},
    'mountain': {'difference': 0.0, 'image': None},
    'sea': {'difference': 0.0, 'image': None},
    'street': {'difference': 0.0, 'image': None},
}

with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        img = inputs[i].view(1, 3, 150, 150)

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
    plt.title(
        f"Predicted: {key}, Actual: {classes[(classes.index(key) + 1) % 6]}, Difference: {value['difference']:.4f}")
    plt.axis('off')
    plt.imshow(value['image'])
    i = i + 1
# %%
plt.savefig('chart.png')