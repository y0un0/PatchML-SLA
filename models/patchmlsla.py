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

class PatchMLSLA(nn.Module):
    def __init__(self, model_name="efficientnet_b4", n_blocks=5, embed_dim=256, n_cls=20):
        super().__init__()
        # Extracting patch embeddings
        if model_name in HANDLED_ENCODER:
            self.encoder = CustomEfficientNet(model_name=model_name, n_blocks=n_blocks, embed_dim=embed_dim)
        else:
            print("Encoder {} is not handled. Try one of these encoders {}".format(model_name, HANDLED_ENCODER))
        # TODO: CodeBook generation
        # Extracting Image embeddings
        self.cross_attn = PatchMLSLAAttention()
        # TODO: Shared Classifier

    def forward(self, patches):
        """
        @param patches: Patches extracted from the original image [num_patches, channel, height, width]
        """
        # [num_patches, channel, height, width] -> [num_patches, embed_dim]
        patch_embs = self.encoder(patches)
        pass

class PatchMLSLAAttention(nn.Module):
    def __init__(self):
        super().__init__()
        pass

class PatchMLSLAMLP(nn.Module):
    def __init__(self):
        super().__init__()
        pass

if __name__ == "__main__":
    model = PatchMLSLA(model_name="efficientnet_b4", n_blocks=4, embed_dim=256, n_cls=10)
    print(model)