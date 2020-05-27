import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Layer definition
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)

        self.fc1 = nn.Linear(in_features=32*4*4, out_features=512)
        self.drop = nn.Dropout()
        self.fc2 = nn.Linear(in_features=512, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # conv1
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        # conv2
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        # fc1
        x = F.relu(self.fc1(self.drop(x.view(-1, 32*4*4))))
        # fc2
        x = F.relu(self.fc2(x))
        # fc3
        x = self.fc3(x)

        return x


# Hyperparameters
num_epochs = 8
num_classes = 10
batch_size = 100
lr = 0.001

# Using the MNIST dataset
trans = transforms.Compose(
    [transforms.ToTensor()])

train_data = datasets.MNIST(root='./data', train=True,
                            download=True, transform=trans)

test_data = datasets.MNIST(root='./data', train=False,
                           download=True, transform=trans)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False)

# Training the network
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

total_step = len(train_loader)
loss_list = []
acc_list = []

for epoch in range(num_epochs):
    i = 0
    print('Epoch: {}'.format(epoch + 1))
    for batch in train_loader:
        images = batch[0].to(device)
        labels = batch[1].to(device)
        preds = model(images)

        loss = F.cross_entropy(preds, labels)
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total = labels.size(0)
        _, predicted = torch.max(preds.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print(' Step: {}/{}    Loss: {:.4f}    Accuracy: {:.2f}%'
                  .format(i + 1, total_step, loss.item(), (correct / total) * 100))

        i += 1
    print()

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy: {:.2f} %'.format(
        (correct / total) * 100))

# Save the model
torch.save(model.state_dict(), './models/model.ckpt')
