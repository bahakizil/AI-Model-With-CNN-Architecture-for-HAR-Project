import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import HMDB51
from torch.utils.data import DataLoader

# Dönüşümler
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),  # Giriş boyutlarına uygun şekilde yeniden boyutlandırma
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Veri yükleyicilerini oluşturma
train_data = HMDB51(root='./data', train=True, download=True, transform=transform)
test_data = HMDB51(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

# Mimari kısmı
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)  # HMDB51 veri seti için giriş boyutu
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 51)  # HMDB51 veri seti için çıkış boyutu

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Kayıp fonksiyonu ve optimizer tanım kısmı
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Eğitim
for epoch in range(5):  # 5 epoch döngüsü
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:  
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

# Test ve doğruluk hesaplama
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the %d test videos: %d %%' % (
    len(test_data), 100 * correct / total))
