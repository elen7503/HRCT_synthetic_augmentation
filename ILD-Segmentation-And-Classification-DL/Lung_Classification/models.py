import torch.nn as nn

class Classifier(nn.Module):
    """
    CNN Classifier for 32x32 CT patches.
    
    Architecture:
        Input: (batch, 1, 32, 32)
        Conv1 → BN → ReLU → MaxPool → (batch, 32,  16, 16)
        Conv2 → BN → ReLU → MaxPool → (batch, 64,   8,  8)
        Conv3 → BN → ReLU → MaxPool → (batch, 128,  4,  4)
        Flatten → FC1(2048→512) → Dropout → FC2(512→num_classes)
    """
    def __init__(self, num_classes=5):
        super(Classifier, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 32 -> 16
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 16 -> 8
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 8 -> 4
        )

        # After 3 MaxPool: 32 → 16 → 8 → 4   =>  128 * 4 * 4 = 2048
        self.fc1  = nn.Linear(128 * 4 * 4, 512)
        self.drop = nn.Dropout(p=0.5)
        self.fc2  = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)   # safe flatten: (batch, 128*4*4)
        x = self.drop(self.fc1(x))
        x = self.fc2(x)
        return x


# AutoEncoder Model For Feature Extraction (**CODE UNDER DEVELOPMENT*)
# Todo: Add a Variational Autoencoder implementation

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        # Decoder layers
        self.deconv1 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(
            32, 16, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(16)
        self.relu5 = nn.ReLU(inplace=True)
        self.deconv3 = nn.ConvTranspose2d(
            16, 1, kernel_size=4, stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.deconv1(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.deconv2(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.deconv3(x)
        x = self.sigmoid(x)
        return x

