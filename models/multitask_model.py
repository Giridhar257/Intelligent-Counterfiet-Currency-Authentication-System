import torch
import torch.nn as nn
import torchvision.models as models

class MultiTaskModel(nn.Module):
    def __init__(self, num_denoms=9, num_countries=2):
        super().__init__()

        # CNN backbone
        self.cnn = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        self.cnn.classifier = nn.Identity()
        cnn_dim = 1280

        # Texture branch
        self.texture_fc = nn.Linear(128, 128)
        
        self.text_fc = nn.Linear(6, 32)

        # Fusion
        self.shared = nn.Sequential(
            nn.Linear(cnn_dim + 128 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Heads
        self.auth_head = nn.Linear(256, 2)              
        self.denom_head = nn.Linear(256, num_denoms)   
        self.country_head = nn.Linear(256, num_countries)  

    def forward(self, x, texture_features, text_features):

        cnn_feat = self.cnn(x)
        texture_feat = self.texture_fc(texture_features)
        text_feat = self.text_fc(text_features)

        combined = torch.cat([cnn_feat, texture_feat,text_feat], dim=1)
        shared = self.shared(combined)

        auth_out = self.auth_head(shared)
        denom_out = self.denom_head(shared)
        country_out = self.country_head(shared)

        return auth_out, denom_out, country_out