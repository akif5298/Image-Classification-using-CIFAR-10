import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations for the images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data',
                                 train=True,
                                 download=True,
                                 transform=transform)
test_dataset = datasets.CIFAR10(root='./data',
                                train=False,
                                download=True,
                                transform=transform)

# Define data loaders
train_loader = DataLoader(train_dataset,
                          batch_size=64,
                          shuffle=True,
                          num_workers=4)
test_loader = DataLoader(test_dataset,
                         batch_size=64,
                         shuffle=False,
                         num_workers=4)


# Define Convolutional Neural Network (CNN) model
class CNN(nn.Module):

  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
    self.fc1 = nn.Linear(32 * 8 * 8, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.pool(self.relu(self.conv1(x)))
    x = self.pool(self.relu(self.conv2(x)))
    x = x.view(-1, 32 * 8 * 8)
    x = self.relu(self.fc1(x))
    x = self.fc2(x)
    return x


# Instantiate CNN model, loss function, and optimizer
cnn_model = CNN().to(device)
criterion_cnn = nn.CrossEntropyLoss()
optimizer_cnn = optim.SGD(cnn_model.parameters(), lr=0.01, momentum=0.9)

# Training loop for CNN
for epoch in range(10):  # Adjust the number of epochs as needed
  cnn_model.train()
  running_loss = 0.0
  for inputs, labels in train_loader:
    inputs, labels = inputs.to(device), labels.to(device)

    optimizer_cnn.zero_grad()
    outputs = cnn_model(inputs)
    loss = criterion_cnn(outputs, labels)
    loss.backward()
    optimizer_cnn.step()

    running_loss += loss.item()

  print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Evaluate CNN model on test set
cnn_model.eval()
correct = 0
total = 0
with torch.no_grad():
  for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = cnn_model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(f"Accuracy of the CNN on the test set: {100 * correct / total}%")
