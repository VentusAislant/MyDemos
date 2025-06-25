from torch import nn


# model: AlexNet
class AlexNet(nn.Module):
    # implement of AlexNet
    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()
        # if input size: torch.Size([batch, 3, 224, 224])
        self.feature_extract = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            # torch.Size([batch, 96, 54, 54])  (224-11+4)//4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # torch.Size([batch, 96, 26, 26])  (54-3+2)//2
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            # torch.Size([batch, 256, 26, 26])  (26-5+2*2+1)//1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # torch.Size([batch, 256, 12, 12])  (26-3+2)//2
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            # torch.Size([batch, 384, 12, 12])  (12-3+1*2+1)//1
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),  # torch.Size([batch, 384, 12, 12])
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),  # torch.Size([batch, 256, 12, 12])
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # torch.Size([batch, 256, 5, 5])
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # # torch.Size([batch, 256*5*5])
            nn.Linear(in_features=256 * 5 * 5, out_features=32 * 5 * 5),  # different from orignial alexnet
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=32 * 5 * 5, out_features=4 * 5 * 5),  # different from orignial alexnet
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=100, out_features=num_classes)  # different from orignial alexnet
        )

        # init_weight
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, X):
        return self.classifier(self.feature_extract(X))
