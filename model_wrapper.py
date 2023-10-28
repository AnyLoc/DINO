# Wrappers to produce final models

# %%
import torch
from torch import nn
# Internal
from dinov2_extractor import DinoV2ExtractFeatures
from utilities import VLAD


# %%
class AnyLocVladDinov2(nn.Module):
    """
        Wrapper around the AnyLoc-VLAD-DINOv2 model in the paper.
        It basically has the DINOv2 ViT feature extraction and the
        VLAD descriptor construction in a single module.
    """
    def __init__(self, c_centers: torch.Tensor, 
                dino_model: str = "dinov2_vitg14", layer: int = 31, 
                facet: str = "value", num_c: int = 32):
        super().__init__()
        # DINOv2 feature extractor
        self.dino_extractor = DinoV2ExtractFeatures(dino_model, layer,
                facet)
        # VLAD clustering
        self.vlad = VLAD(num_c)
        self.vlad.c_centers = c_centers # Load cluster centers
        self.vlad.fit(None) # Load the database (vocabulary/c_centers)
    
    # Forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_pt = x
        # TODO: Add batching feature
        assert len(img_pt.shape) == 4 and img_pt.shape[0] == 1 and \
                img_pt.shape[1] == 3, "Pass only single RGB image"
        # Extract features
        ret = self.dino_extractor(img_pt)
        gd = self.vlad.generate(ret[0])
        return gd
