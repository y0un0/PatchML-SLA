from models import CustomEfficientNet, VanillaCNN
from utils import MultiResolutionPatches
from einops import rearrange
import torch
import torch.nn as nn

HANDLED_ENCODER = ["efficientnet_b0",
                   "efficientnet_b1",
                   "efficientnet_b2",
                   "efficientnet_b3",
                   "efficientnet_b4",
                   "efficientnet_b5",
                   "efficientnet_b6",
                   "efficientnet_b7", 
                   "vanilla-cnn"]

class PatchMLSL(nn.Module):
    def __init__(self, 
                model_name="efficientnet_b4", 
                n_blocks=4, 
                intermediate_dim=128, 
                embed_dim=256, 
                n_cls=20,
                patch_size=64,
                stride=64,
                num_resolutions=3,
                downsample_ratio=2,
                interpolation="bilinear"):
        super().__init__()

        # Initialize patch extractor
        self.patch_extractor = MultiResolutionPatches(patch_size=patch_size, stride=stride, num_resolutions=num_resolutions, 
                                                    downsample_ratio=downsample_ratio, interpolation=interpolation)
        # Extracting patch embeddings
        if model_name in HANDLED_ENCODER and "vanilla" not in model_name:
            self.encoder = CustomEfficientNet(model_name=model_name, n_blocks=n_blocks, embed_dim=embed_dim)
        elif model_name in HANDLED_ENCODER and "vanilla" in model_name:
            self.encoder = VanillaCNN(embed_dim=embed_dim)
        else:
            print("Encoder {} is not handled. Try one of these encoders {}".format(model_name, HANDLED_ENCODER))

        # Extracting Image embeddings
        self.cross_attn = PatchMLSLAttention(n_cls=n_cls, embed_dim=embed_dim)
        self.mlp = PatchMLSLMLP(embed_dim)
        #Shared classifier
        self.classifier = PatchMLSLClassifier(n_cls=n_cls, embed_dim=embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, images):
        """
        @param images: Input images [batch_size, channel, height, width]
        @return classifier: Vector of the most probable classes [num_classes]
        """
        # Divide the image into patches
        patches = self.patch_extractor.extract_patches(images) 
        # Encode patches
        # From [batch_size * num_patches, channels, height, width] to [batch_size * num_patches, embed_dim]
        patch_embs = self.encoder(patches)
        # Normalize the patch embeddings
        patch_embs = self.norm(patch_embs)
        # Reshape embeddings back into batch form
        # From [batch_size * num_patches, embed_dim] to [batch_size, num_patches, embed_dim]
        patch_embs = rearrange(patch_embs, '(b np) d -> b np d', b=images.size(0))
        # Apply cross attention
        # [batch_size, num_patches, embed_dim] -> [batch_size, embed_dim]
        patches_attn = self.cross_attn(patch_embs)
        # Image representation with MLP
        # [batch_size, embed_dim] -> [batch_size, embed_dim]
        mlp = self.mlp(patches_attn)
        image_repr = patches_attn + mlp  # Combine attention and MLP output
        # Normalize image representation
        image_repr = self.norm(image_repr)
        # Classification
        # [batch_size, embed_dim] -> [batch_size, num_classes]
        classifier = self.classifier(image_repr)
        return classifier, image_repr
    
class PatchMLSLAttention(nn.Module):
    def __init__(self, n_cls=20, embed_dim=256):
        super().__init__()
        tensor_codebook = torch.randn(n_cls, embed_dim)
        nn.init.xavier_uniform_(tensor_codebook)
        self.codebook = nn.Parameter(tensor_codebook, requires_grad=True)

    def forward(self, patch_embed):
        # Compute Attention weights (Matrix A)
        attn_weights = torch.matmul(self.codebook, patch_embed.transpose(1, 2))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        # Perform the weighted sum (attn_weights * patch_embed)
        attn_patch = torch.matmul(attn_weights, patch_embed)
        return attn_patch
    
class PatchMLSLMLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, attn_patch):
        hidden_state = self.fc1(attn_patch)
        hidden_state = nn.functional.gelu(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state
    
class PatchMLSLClassifier(nn.Module):
    def __init__(self, n_cls=20, embed_dim=256):
        super().__init__()
        tensor_cls_weights = torch.randn(n_cls, embed_dim)
        nn.init.xavier_uniform_(tensor_cls_weights)
        self.cls_weights = nn.Parameter(tensor_cls_weights, requires_grad=True)
    
    def forward(self, image_rep):
        logits = torch.matmul(self.cls_weights, image_rep.transpose(1, 2))
        # logits = (image_rep * self.cls_weights.unsqueeze(0)).sum(dim=-1)
        # logits = torch.einsum("bld,ld->bl", image_rep, self.cls_weights)
        # Apply softmax
        logits = nn.functional.softmax(logits, dim=-1)
        # Extracting the diagonal
        output = logits.diagonal(dim1=-2, dim2=-1)
        return output
        

if __name__ == "__main__":
    model = PatchMLSL(model_name="efficientnet_b4", n_blocks=4, intermediate_dim=128, embed_dim=256, n_cls=3)
    input_tensor = torch.rand(2, 3, 640, 640)
    output = model(input_tensor)
    print(output[0].shape, output[1].shape)