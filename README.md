# MNIST com Redes Neurais

Criando uma rede neural simples para classificar imagens de dígitos escritos à mão do conjunto de dados MNIST.

---

## Importação de Bibliotecas

```python
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from plotly import express as px
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
```

---

## Carregamento e Pré-processamento dos Dados

```python
# Transformação dos dados para tensores
transform = transforms.ToTensor()

# Carregamento do dataset de treino
trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Carregamento do dataset de validação
valset = datasets.MNIST('./MNIST_data/', download=True, train=False, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
```

---

## Visualização dos Dados

```python
# Extração de um batch de imagens
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Exibição da primeira imagem do batch
px.imshow(images[0].squeeze().numpy(), color_continuous_scale='gray').show()

# Verificação das dimensões da imagem
print(images[0].shape)  # Saída esperada: torch.Size([1, 28, 28])
```

---

## Implementação da Rede Neural

### Arquitetura do Modelo
```python
class Modelo(nn.Module):
    def __init__(self):
        super(Modelo, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
```

---

## Funções de Treino e Validação

### Treino
```python
def treino(modelo, trainloader, device, epochs=10):
    inicio = time()
    otimizador = optim.SGD(modelo.parameters(), lr=0.01, momentum=0.5)
    criterio = nn.NLLLoss()
    modelo.train()

    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)
            otimizador.zero_grad()
            output = modelo(images.to(device))
            loss = criterio(output, labels.to(device))
            loss.backward()
            otimizador.step()
            running_loss += loss.item()
        print(f"Epoch {e} - Training loss: {running_loss/len(trainloader)}")
    
    print(f"Tempo de treino: {time() - inicio}")
```

### Validação
```python
def validacao(modelo, valloader, device):
    modelo.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in valloader:
            images = images.view(images.shape[0], -1)
            output = modelo(images.to(device))
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
    
    print(f"Acurácia: {100 * correct / total}%")
```

---

## Execução do Treinamento

```python
# Inicialização do modelo e dispositivo (GPU/CPU)
modelo = Modelo()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo.to(device)

# Treino por 30 épocas
treino(modelo, trainloader, device, epochs=30)

# Validação do modelo
validacao(modelo, valloader, device)  # Acurácia esperada: ~95-97%
```

---

## Salvamento e Carregamento do Modelo

```python
# Salvamento dos pesos do modelo
torch.save(modelo.state_dict(), 'modelo.pth')

# Carregamento do modelo salvo
modelo_carregado = Modelo()
modelo_carregado.load_state_dict(torch.load('modelo.pth'))
```

---

## Exemplo de Previsão

```python
# Seleção de uma imagem do dataset
img = images[0]
numero = img.view(1, 28*28)

# Exibição da imagem
px.imshow(img.squeeze().numpy(), color_continuous_scale='gray').show()

# Previsão com o modelo treinado
with torch.no_grad():
    logps = modelo(numero.to(device))
    ps = torch.exp(logps)
    probab = list(ps.cpu().numpy()[0])
    print(f"Previsão: {probab.index(max(probab))}")  # Exemplo de saída: "Previsão: 5"
```

---

## Conclusão
Este projeto demonstra a implementação de uma rede neural simples para classificação do MNIST, alcançando alta acurácia. O modelo utiliza camadas totalmente conectadas e funções de ativação ReLU, com otimização via SGD.