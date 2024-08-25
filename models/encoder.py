import torch
import torch.nn as nn
import timm

class CustomEfficientNet(nn.Module):
    def __init__(self, model_name, n_blocks, embed_dim=256):
        super().__init__()
        self.n_blocks = n_blocks
        base_model = timm.create_model(model_name=model_name, pretrained=False)
        # Constructing new shared encoder
        self.model = self.construct_model(base_model)
        del base_model

        # Adding the average pooling and fc layer
        layers_dim = [dim for dim in self.model.parameters()]
        last_layer_dim = layers_dim[-1].size()[-1]
        self.model.avg_pool = nn.AdaptiveAvgPool2d(last_layer_dim)
        self.model.fc = nn.Linear(in_features=last_layer_dim, out_features=embed_dim)

    def extract_block(self, base_model):
        return nn.Sequential(*[base_model.blocks[i] for i in range(self.n_blocks + 1)])
    
    def construct_model(self, base_model):
        model = nn.Sequential()
        model.conv_stem = base_model.conv_stem
        model.bn1 = base_model.bn1
        model.blocks = self.extract_block(base_model)
        return model
    
    def forward(self, x):
        features = self.model(x)
        return features
    
if __name__ == "__main__":
    model = CustomEfficientNet("efficientnet_b4", n_blocks=5)
    input_tensor = torch.rand(1, 3, 224, 224)
    output = model(input_tensor)
    print(output)