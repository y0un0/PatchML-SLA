from encoder import *
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
    def __init__(self, model_name="efficientnet_b4", n_blocks=5, embed_dim=256, n_cls=20):
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
        # TODO: Shared Classifier

    def forward(self, patches):
        """
        @param patches: Patches extracted from the original image [num_patches, channel, height, width]
        """
        # [num_patches, channel, height, width] -> [num_patches, embed_dim]
        patch_embs = self.encoder(patches)
        pass

class CodebookLabel(nn.Module):
    def __init__(self,  n_cls=20, embed_dim=256):
        super().__init__()
        self.codebook = nn.Embedding(n_cls, embed_dim)
    
    def forward(self):
        return self.codebook
    
class PatchMLSLAttention(nn.Module):
    def __init__(self):
        super().__init__()
        pass

class PatchMLSLMLP(nn.Module):
    def __init__(self):
        super().__init__()
        pass

if __name__ == "__main__":
    model = PatchMLSL(model_name="efficientnet_b4", n_blocks=4, embed_dim=256, n_cls=10)
    print(model)