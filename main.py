# Execute este código para verificar se o dataset é carregado corretamente
import torch
from torchvision import datasets, transforms

transform = transforms.ToTensor()
trainset = datasets.MNIST(root='./MNIST_data/', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = next(dataiter)
print(images.shape)  # Deve exibir torch.Size([64, 1, 28, 28])
print(labels.shape)  # Deve exibir torch.Size([64])