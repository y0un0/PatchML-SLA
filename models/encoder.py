import torch
import torch.nn as nn
import timm

class CustomEfficientNet(nn.Module):
    def __init__(self, model_name, n_blocks, embed_dim=256):
        super().__init__()
        self.n_blocks = n_blocks
        self.base_model = timm.create_model(model_name=model_name, pretrained=False)
        # Constructing new shared encoder
        self.conv_stem = self.base_model.conv_stem
        self.bn1 = self.base_model.bn1
        self.blocks = self.extract_block()
        del self.base_model

        # Adding the average pooling and fc layer
        layers_dim = [dim for dim in self.blocks.parameters()]
        last_layer_dim = layers_dim[-1].size()[-1]
        self.avg_pool = nn.AdaptiveAvgPool2d(last_layer_dim)
        self.fc = nn.Linear(in_features=last_layer_dim, out_features=embed_dim)

    def extract_block(self):
        return nn.Sequential(*[self.base_model.blocks[i] for i in range(self.n_blocks + 1)])
    
    def forward(self, x):
        conv_stem = self.conv_stem(x)
        bn1 = self.bn1(conv_stem)
        features = self.blocks(bn1)
        return features
    
if __name__ == "__main__":
    model = CustomEfficientNet("efficientnet_b4", n_blocks=5)
    input_tensor = torch.rand(1, 3, 224, 224)
    output = model(input_tensor)
    for name, param in model.named_parameters():
        print(name, param.size())