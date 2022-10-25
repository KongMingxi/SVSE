import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

from .reset import resnet18, resnet50


def build_model_training(cfg):

    # configs for network
    if cfg.TRAINER.BASELINE.ONLY:
        model = BaselineModel(cfg)

    return model


class BaselineModel(BaseModel):

    def __init__(self, cfg):
        super().__init__()
        
        # set up backbone
        if cfg.MODEL.BACKBONE.NAME == 'resnet18':
            self.backbone = resnet18(x)
        elif cfg.MODEL.BACKBONE.NAME == 'resnet50':
            self.backbone = resnet50(x)
        else:
            print('A model name is required')

        fdim = self.backbone.out_features
        self._fdim = fdim

        # set up classifier
        num_classes = cfg.DATASET.NUM_CLASSES
                
        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Linear(fdim, num_classes)

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, return_feature=False):
        f = self.backbone(x)

        y_logit = classifier(f)

        return y_logit

    def return_feature():
        f = self.backbone(x)

        return f


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
