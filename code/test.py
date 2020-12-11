'''import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
import matplotlib
matplotlib.use("TkAgg")

device = torch.device("cpu")
classes = ["cat", "dog", "elephant", "panda"]

model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 4)

model_ft.load_state_dict(torch.load(f=r"F:\EmbeddedProject\Classification\classifier.pth", map_location=torch.device('cpu')))

img = cv2.imread(r"F:\EmbeddedProject\Classification\DATASET\test\panda\panda_00017.jpg")
img1 = cv2.resize(img,(224, 224))
img2 = torch.tensor(img1)
img3 = torch.reshape(img2, (1, 3, 224, 224))

out = model_ft(img3.float())
out = list(out.detach().numpy()[0])

plt.imshow(img, cmap='gray')
plt.title(classes[out.index(max(out))])
plt.show()'''
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as data
import torchvision.models as models


def DATA_LOADER():
    #train_folder = r"F:\EmbeddedProject\Classification\DATASET\train"
    #test_folder = r"F:\EmbeddedProject\Classification\DATASET\test"
    BATCH_SIZE = 10
    Transforming = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation((30, 120)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.4, 0.5], std=[0.22, 0.24, 0.22])])
    train_data = torchvision.datasets.ImageFolder(root=r"F:\EmbeddedProject\Classification\DATASET\train", transform=Transforming)
    test_data = torchvision.datasets.ImageFolder(root=r"F:\EmbeddedProject\Classification\DATASET\test", transform=Transforming)
    train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader, test_loader

train_loader, test_loader = DATA_LOADER()
dataiter = iter(test_loader)
images, labels = dataiter.next()

model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 4)

model_ft.load_state_dict(torch.load(f=r"F:\EmbeddedProject\Classification\classifier.pth", map_location=torch.device('cpu')))

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model_ft(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on test images: %d %%' % (100 * correct / total))
