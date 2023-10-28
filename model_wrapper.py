# Wrappers to produce final models

# %%
import torch
from torch import nn
import einops as ein
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
                facet: str = "value", num_c: int = 32, 
                device: torch.device = "cpu"):
        super().__init__()
        # DINOv2 feature extractor
        self.dino_model = dino_model
        self.layer = layer
        self.facet = facet
        self.device = torch.device(device)
        self.dino_extractor = self._get_dino_extractor()
        # VLAD clustering
        self.vlad = VLAD(num_c)
        self.vlad.c_centers = c_centers # Load cluster centers
        self.vlad.fit(None) # Load the database (vocabulary/c_centers)
    
    # Extractor
    def _get_dino_extractor(self):
        return DinoV2ExtractFeatures(
            dino_model=self.dino_model, layer=self.layer, 
            facet=self.facet, device=self.device)
    
    # Move DINO model to device
    def to(self, device: torch.device):
        self.device = torch.device(device)
        self.dino_extractor = self._get_dino_extractor()
    
    # Wrapper around CUDA
    def cuda(self):
        self.to("cuda")
    
    # Forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_pt = x
        shapes = ein.parse_shape(img_pt, "b c h w")
        assert shapes["c"] == 3, "Image(s) must be RGB!"
        assert shapes["h"] % 14 == shapes["w"] % 14 == 0, \
                "Height and width should be multiple of 14 (for "\
                "patching)"
        img_pt = img_pt.to(self.device)
        # Extract features
        ret = self.dino_extractor(img_pt)   # [b, (H.W), dino_dim]
        gds = torch.empty([shapes["b"], # Global descs: [b, vlad_dim]
                self.vlad.desc_dim * self.vlad.num_clusters])
        for i in range(shapes["b"]):
            gds[i] = self.vlad.generate(ret[i].cpu())   # VLAD on CPU!
        return gds.to(self.device)
