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


# Define Artificial Neural Network (ANN) model
class ANN(nn.Module):

  def __init__(self):
    super(ANN, self).__init__()
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(32 * 32 * 3, 128)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x


# Instantiate ANN model, loss function, and optimizer
ann_model = ANN().to(device)
criterion_ann = nn.CrossEntropyLoss()
optimizer_ann = optim.SGD(ann_model.parameters(), lr=0.01, momentum=0.9)

# Training loop for ANN
for epoch in range(20):  # Adjust the number of epochs as needed
  ann_model.train()
  running_loss = 0.0
  for inputs, labels in train_loader:
    inputs, labels = inputs.to(device), labels.to(device)

    optimizer_ann.zero_grad()
    outputs = ann_model(inputs)
    loss = criterion_ann(outputs, labels)
    loss.backward()
    optimizer_ann.step()

    running_loss += loss.item()

  print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Evaluate ANN model on test set
ann_model.eval()
correct = 0
total = 0
with torch.no_grad():
  for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = ann_model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(f"Accuracy of the ANN on the test set: {100 * correct / total}%")
