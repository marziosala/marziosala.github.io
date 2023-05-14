---
layout: splash
permalink: /dirty-documents/
title: "Denoising Autoencoders"
header:
  overlay_image: /assets/images/dirty-documents/dirty-documents-splash.jpeg
excerpt: "An application of denoising autoencoders to the dirty document dataset."
---

```python
import random
from pathlib import Path
import torch
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from torch import nn
from torch import optim
from collections import OrderedDict

from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
```

    c:\Users\dragh\miniconda3\envs\torch\lib\site-packages\torchvision\io\image.py:11: UserWarning: Failed to load image Python extension: [WinError 127] Impossibile trovare la procedura specificata
      warn(f"Failed to load image Python extension: {e}")
    


```python
random.seed(42)
torch.manual_seed(43);
```


```python
data_dir = Path('./data')
train_dir = data_dir / 'train'
train_cleaned_dir = data_dir / 'train_cleaned'
test_dir = data_dir / 'test'

train_images = sorted(train_dir.glob('*.png'))
train_cleaned_images = sorted(train_cleaned_dir.glob('*.png'))
test_images = sorted(test_dir.glob('*.png'))

print('Number of Images in train:', len(train_images))
print('Number of Images in train_cleaned:', len(train_cleaned_images))
print('Number of Images in test:', len(test_images))
```

    Number of Images in train: 144
    Number of Images in train_cleaned: 144
    Number of Images in test: 72
    


```python
transform = transforms.Compose([
    transforms.Resize((320, 480)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

X = []
for image in train_images:
    pil_image = Image.open(image)
    pil_image = transform(pil_image)
    X.append(pil_image)
    
Y = []
for image in train_cleaned_images:
    pil_image = Image.open(image)
    pil_image = transform(pil_image)
    Y.append(pil_image)
    
test_images_transformed = []
for image in test_images:
    pil_image = Image.open(image)
    pil_image = transform(pil_image)
    test_images_transformed.append(pil_image)
```


```python
def imshow(image, ax=None, title):
    if ax is None:
        fig, ax = plt.subplots()
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # undo preprocessing
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    image = std * image + mean
    
    # image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    ax.grid(False)
    ax.set_title(title)
```


```python
dataset = [(X[i], Y[i]) for i in range(len(X))]
random.shuffle(dataset)

split_size = 0.9
index = int(len(dataset)*split_size)

train_dataset = dataset[:index]
valid_dataset = dataset[index:]

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)
```


```python
images, targets = next(iter(train_loader))
images, targets = images.numpy(), targets.numpy()

def show_pair(i):
    plt.figure(figsize=(12, 14))
    ax = plt.subplot(1, 2, 1)
    imshow(images[i], ax, 'Original Image')
    ax = plt.subplot(1, 2, 2)
    imshow(targets[i], ax, 'Denoised Image')
```


```python
for i in range(20):
    show_pair(i)
    plt.show()
```


    
![png](/assets/images/dirty-documents/dirty-documents-1.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-2.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-3.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-4.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-5.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-6.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-7.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-8.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-9.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-10.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-11.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-12.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-13.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-14.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-15.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-16.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-17.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-18.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-19.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-20.png)
    



```python
class ConvDenoiser(nn.Module):
    
    def __init__(self):
        super().__init__();
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2,2)
        
        self.convt_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.convt_2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.convt_3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = F.relu(self.convt_1(x))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.convt_2(x))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.sigmoid(self.convt_3(x))
        
        return x
```


```python
model = ConvDenoiser()
print(model)
```

    ConvDenoiser(
      (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (convt_1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (convt_2): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (convt_3): Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    


```python
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
model = model.to(device)
```


```python
def train(model, train_loader, valid_loader, num_epochs):
    
    for epoch in range(num_epochs):
        training_loss = 0.0        
        for images, targets in train_loader:            
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item()
        
        with torch.no_grad():
            valid_loss = 0
            for images, targets in valid_loader:
                images, targets = images.to(device), targets.to(device)
            
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                valid_loss += loss.item()
        
        print(f'Epoch: {epoch + 1}/{num_epochs}    Training Loss: {training_loss/len(train_loader):.3f}    ' \
              f'Testing Loss: {valid_loss/len(valid_loader):.3f}')
```


```python
train(model, train_loader, valid_loader, 20)
torch.save(model.state_dict(), './mode.pt')
```

    c:\Users\dragh\miniconda3\envs\torch\lib\site-packages\torch\nn\functional.py:1960: UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
      warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
    

    Epoch: 1/20    Training Loss: 0.270    Testing Loss: 0.226
    Epoch: 2/20    Training Loss: 0.225    Testing Loss: 0.211
    Epoch: 3/20    Training Loss: 0.204    Testing Loss: 0.191
    Epoch: 4/20    Training Loss: 0.178    Testing Loss: 0.162
    Epoch: 5/20    Training Loss: 0.158    Testing Loss: 0.150
    Epoch: 6/20    Training Loss: 0.148    Testing Loss: 0.147
    Epoch: 7/20    Training Loss: 0.145    Testing Loss: 0.137
    Epoch: 8/20    Training Loss: 0.138    Testing Loss: 0.131
    Epoch: 9/20    Training Loss: 0.129    Testing Loss: 0.125
    Epoch: 10/20    Training Loss: 0.123    Testing Loss: 0.120
    Epoch: 11/20    Training Loss: 0.118    Testing Loss: 0.115
    Epoch: 12/20    Training Loss: 0.113    Testing Loss: 0.112
    Epoch: 13/20    Training Loss: 0.109    Testing Loss: 0.107
    Epoch: 14/20    Training Loss: 0.105    Testing Loss: 0.103
    Epoch: 15/20    Training Loss: 0.101    Testing Loss: 0.100
    Epoch: 16/20    Training Loss: 0.097    Testing Loss: 0.097
    Epoch: 17/20    Training Loss: 0.094    Testing Loss: 0.096
    Epoch: 18/20    Training Loss: 0.092    Testing Loss: 0.093
    Epoch: 19/20    Training Loss: 0.090    Testing Loss: 0.091
    Epoch: 20/20    Training Loss: 0.088    Testing Loss: 0.088
    


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    ~\AppData\Local\Temp/ipykernel_6600/4232275783.py in <module>
          1 train(model, train_loader, valid_loader, 20)
    ----> 2 model.save(model.state_dict(), './model.pt')
    

    c:\Users\dragh\miniconda3\envs\torch\lib\site-packages\torch\nn\modules\module.py in __getattr__(self, name)
       1205             if name in modules:
       1206                 return modules[name]
    -> 1207         raise AttributeError("'{}' object has no attribute '{}'".format(
       1208             type(self).__name__, name))
       1209 
    

    AttributeError: 'ConvDenoiser' object has no attribute 'save'



```python
# model.load_state_dict(torch.load('./model.pt'))
```




    <All keys matched successfully>




```python
#testing on test set
for i in range(10):
    image = test_images_transformed[i]
    image = image.unsqueeze(0).to(device)
    output = model(image)

    image, output = image.detach().cpu().numpy(), output.detach().cpu().numpy()

    plt.figure(figsize=(12,14))
    ax = plt.subplot(1,2,1)
    imshow(image[0], ax, 'Original Image')

    ax = plt.subplot(1,2,2)
    imshow(output[0], ax, 'Denoised Image')
    
    plt.show()
```

    c:\Users\dragh\miniconda3\envs\torch\lib\site-packages\torch\nn\functional.py:1960: UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
      warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")
    


    
![png](/assets/images/dirty-documents/dirty-documents-21.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-22.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-23.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-24.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-25.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-26.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-27.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-28.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-29.png)
    



    
![png](/assets/images/dirty-documents/dirty-documents-30.png)
    

