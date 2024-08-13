Demo Link : https://youtu.be/QUruhz7mr1c 

## Model summary

```python
import torch.nn as nn

class CNNModel4(nn.Module):
    def __init__(self):
        super(CNNModel4, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.leaky_relu1 = nn.LeakyReLU(0.1)  # Leaky ReLU activation
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.leaky_relu2 = nn.LeakyReLU(0.1)  # Leaky ReLU activation
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.leaky_relu3 = nn.LeakyReLU(0.1)  # Leaky ReLU activation
        self.fc2 = nn.Linear(512, 100)  # Output 100 classes for coarse labels

    def forward(self, x):
        x = self.pool1(self.leaky_relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.leaky_relu2(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = self.leaky_relu3(self.fc1(x))
        x = self.fc2(x)
        return x
```

### Model Architecture

- **Convolution Layers:**
  - `self.conv1`: 32 filters, kernel size 3x3, input channels 3 (RGB), output channels 32.
  - `self.conv2`: 64 filters, kernel size 3x3, input channels 32, output channels 64.

- **Batch Normalization Layers:**
  - `self.bn1`: Applied after `self.conv1`.
  - `self.bn2`: Applied after `self.conv2`.

- **Activation Functions:**
  - Leaky Rectified Linear Unit (Leaky ReLU):
    - `self.leaky_relu1`
    - `self.leaky_relu2`
    - `self.leaky_relu3`

- **Pooling Layers:**
  - `self.pool1`: Kernel size 2x2 and stride of 2, applied after `self.leaky_relu1`.
  - `self.pool2`: Kernel size 2x2 and stride of 2, applied after `self.leaky_relu2`.

- **Dense Layers (Fully Connected):**
  - `self.fc1`: Input features 64x8x8, output features 512.
  - `self.fc2`: Input features 512, output features 100.

### Training Details

- **Optimizer:** Stochastic Gradient Descent (SGD)
- **Loss Function:** Cross Entropy Loss

- **Hyperparameters:**
  - Learning Rate: 0.01
  - Momentum: 0.9
  - Epochs: 9

### Performance

- **Validation Accuracy:** 62.615% (sub-training set and validation set)
- **Test Accuracy:** 83.97% (full training dataset)

### Benchmark

- **Rank:** 2


 
