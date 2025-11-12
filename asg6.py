import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Data transformations
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
data_dir = '.'  # current directory containing train/valid/test
batch_size = 32

train_ds = datasets.ImageFolder(data_dir + '/train', transform=train_tfms)
valid_ds = datasets.ImageFolder(data_dir + '/valid', transform=test_tfms)
test_ds = datasets.ImageFolder(data_dir + '/test', transform=test_tfms)

trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
validloader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
testloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

print(f"Number of training images: {len(train_ds)}")
print(f"Number of validation images: {len(valid_ds)}")
print(f"Number of test images: {len(test_ds)}")
print(f"Number of classes: {len(train_ds.classes)}")
model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

n_inputs = model.classifier[6].in_features
n_classes = len(train_ds.classes)

model.classifier[6] = nn.Sequential(
    nn.Linear(n_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, n_classes),
    nn.LogSoftmax(dim=1)
)

# Enable gradients for classifier parameters
for param in model.classifier[6].parameters():
    param.requires_grad = True

print(model)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier[6].parameters())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Using device: {device}")
n_epochs = 10

for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(trainloader.dataset)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total

    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}, Val Accuracy: {val_acc:.4f}")
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
print(f"Test Accuracy: {test_acc:.4f}")
