import torch
import torch.nn as nn
import timm

class CustomEfficientNet(nn.Module):
    def __init__(self, model_name, n_blocks, adavgpool_size=5, embed_dim=256):
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
        self.avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(adavgpool_size),
                                      nn.Flatten(1))
        self.fc = nn.Linear(in_features=last_layer_dim*adavgpool_size**2, out_features=embed_dim)

    def extract_block(self):
        return nn.Sequential(*[self.base_model.blocks[i] for i in range(self.n_blocks + 1)])
    
    def forward(self, x):
        conv_stem = self.conv_stem(x)
        bn1 = self.bn1(conv_stem)
        features = self.blocks(bn1)
        features = self.avg_pool(features)
        features = self.fc(features)
        return features
    
if __name__ == "__main__":
    model = CustomEfficientNet("efficientnet_b4", n_blocks=5)
    input_tensor = torch.rand(2, 3, 3, 224, 224)
    # Reshape the input
    batch_size = input_tensor.size(0)
    num_patches = input_tensor.size(1)
    reshaped_input_tensor = input_tensor.view(batch_size * num_patches, input_tensor.size(2), input_tensor.size(3), input_tensor.size(4))
    print(reshaped_input_tensor.size())
    output = model(reshaped_input_tensor)
    output = output.view(batch_size, num_patches, -1)
    print(output.size())