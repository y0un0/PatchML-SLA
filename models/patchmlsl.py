from models import CustomEfficientNet
import torch
import torch.nn as nn

HANDLED_ENCODER = ["efficientnet_b0",
                   "efficientnet_b1",
                   "efficientnet_b2",
                   "efficientnet_b3",
                   "efficientnet_b4",
                   "efficientnet_b5",
                   "efficientnet_b6",
                   "efficientnet_b7"]

class PatchMLSL(nn.Module):
    def __init__(self, model_name="efficientnet_b4", n_blocks=5, intermediate_dim=128, embed_dim=256, n_cls=20):
        super().__init__()
        # Extracting patch embeddings
        if model_name in HANDLED_ENCODER:
            self.encoder = CustomEfficientNet(model_name=model_name, n_blocks=n_blocks, embed_dim=embed_dim)
        else:
            print("Encoder {} is not handled. Try one of these encoders {}".format(model_name, HANDLED_ENCODER))
        # Generating the label codebook (row = n_cls, column = embed_dim)
        self.codebook = CodebookLabel(n_cls=n_cls, embed_dim=embed_dim)
        # Extracting Image embeddings
        self.cross_attn = PatchMLSLAttention()
        self.mlp = PatchMLSLMLP(intermediate_dim, embed_dim)
        #Shared classifier
        self.classifier = PatchMLSLClassifier(n_cls=n_cls, embed_dim=embed_dim)

    def forward(self, patches):
        """
        @param patches: Patches extracted from the original image [batch_size, num_patches, channel, height, width]
        @return classifier: Vector of the most probable classes [num_classes]
        """
        # [batch_size, num_patches, channel, height, width] -> [batch_size*num_patches, channel, height, width]
        batch_size = patches.size(0)
        num_patches = patches.size(1)
        patches = patches.view(batch_size * num_patches, patches.size(2), patches.size(3), patches.size(4))
        # [batch_size*num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        patch_embs = self.encoder(patches)
        patch_embs = patch_embs.view(batch_size, num_patches, -1)
        # Patches attention 
        codebook = self.codebook.codebook
        patches_attn = self.cross_attn(codebook, patch_embs)
        # Image representation -> 
        mlp = self.mlp(patches_attn)
        image_repr = patches_attn + mlp
        classifier = self.classifier(image_repr)
        return classifier

class CodebookLabel(nn.Module):
    def __init__(self,  n_cls=20, embed_dim=256):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(n_cls, embed_dim))
    
    def forward(self):
        return self.codebook
    
class PatchMLSLAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, codebook, patch_embed):
        # Compute Attention weights (Matrix A)
        attn_weights = torch.matmul(codebook, patch_embed.transpose(1, 2))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        # Perform the weighted sum (attn_weights * patch_embed)
        attn_patch = torch.matmul(attn_weights, patch_embed)
        return attn_patch
    
class PatchMLSLMLP(nn.Module):
    def __init__(self, intermediate_dim, embed_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, embed_dim)
    
    def forward(self, attn_patch):
        hidden_state = self.fc1(attn_patch)
        hidden_state = nn.functional.relu(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state
    
class PatchMLSLClassifier(nn.Module):
    def __init__(self, n_cls=20, embed_dim=256):
        super().__init__()
        self.cls_weights = nn.Parameter(torch.randn(n_cls, embed_dim))
    
    def forward(self, image_rep):
        logits = torch.matmul(self.cls_weights, image_rep.transpose(1, 2))
        # Apply softmax
        logits = torch.softmax(logits, dim=-1)
        # Extracting the diagonal
        output = logits.diagonal(dim1=-2, dim2=-1)
        return output
        

if __name__ == "__main__":
    model = PatchMLSL(model_name="efficientnet_b4", n_blocks=4, intermediate_dim=128, embed_dim=256, n_cls=3)
    input_tensor = torch.rand(2, 3, 3, 64, 64)
    output = model(input_tensor)
    print(output)