import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class CustomEfficientNet(nn.Module):
    def __init__(self, model_name, n_blocks, adavgpool_size=4, embed_dim=256):
        super().__init__()
        self.n_blocks = n_blocks
        self.base_model = timm.create_model(model_name=model_name, pretrained=False)
        # Constructing new shared encoder
        self.conv_stem = self.base_model.conv_stem
        self.bn1 = self.base_model.bn1
        self.blocks = self.extract_block()
        del self.base_model

        # Adding the average pooling and fc layer
        last_layer_dim = self.blocks[-1][-1].bn3.num_features
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

class VanillaCNN(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.pool = nn.AdaptiveAvgPool2d(5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.fc = nn.Linear(in_features=64*5**2, out_features=embed_dim)
    
    def forward(self, x):
        # Pass through conv1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        # Pass through conv2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        # Pass through fc layer
        x = self.fc(x)
        return x

    
if __name__ == "__main__":
    vanilla = False
    if not vanilla:
        model = CustomEfficientNet("efficientnet_b0", n_blocks=4)
        for name, parameters in model.named_parameters():
            print(name)
        
    else:
        model = VanillaCNN(embed_dim=256)
    # input_tensor = torch.rand(2, 3, 3, 64, 64)
    # # Reshape the input
    # batch_size = input_tensor.size(0)
    # num_patches = input_tensor.size(1)
    # reshaped_input_tensor = input_tensor.view(batch_size * num_patches, input_tensor.size(2), input_tensor.size(3), input_tensor.size(4))
    # print(reshaped_input_tensor.size())
    # output = model(reshaped_input_tensor)
    # output = output.view(batch_size, num_patches, -1)
    # print(output.size())