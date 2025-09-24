import torch
from torchvision import models


class EfficientNetWithEmbeddings(torch.nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.base = models.efficientnet_b0(weights=None)
        self.feature_extractor = self.base.features
        self.pool = self.base.avgpool
        self.embedding_layer = torch.nn.Flatten()
        in_features = self.base.classifier[1].in_features
        self.fc = torch.nn.Linear(in_features, num_classes)

    def forward(self, x, return_embedding=False):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = self.embedding_layer(x)
        out = self.fc(x)

        if return_embedding:
            return out, x

        return out

    def load(self, path):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)

        self.load_state_dict(state_dict)
        self.eval()
        return self
