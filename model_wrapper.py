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
        Wrapper around the AnyLoc-VLAD-DINOv2 model in the paper for
        the domain vocabularies (default).
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
        self.vlad.c_centers = c_centers.to(self.device)
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
        ret = self.dino_extractor(img_pt)   # [b, (nH*nW), dino_dim]
        gds = self.vlad.generate_multi(ret)
        return gds.to(self.device)


# %%
class AnyLocVladNoCacheDinov2(nn.Module):
    """
        Wrapper around the AnyLoc-VLAD-DINOv2 model without the VLAD
        cluster centers. This is useful for using DINOv2 as a feature 
        extractor, and then using VLAD for the clustering.
        If you want to use a cache (cluster centers already computed),
        then use `AnyLocVladDinov2` class instead.
    """
    def __init__(self, dino_model: str = "dinov2_vitg14", 
                layer: int = 31, facet: str = "value", 
                num_c: int = 32, device: torch.device = "cpu")\
                    -> None:
        super().__init__()
        # DINOv2 feature extractor
        self.dino_model = dino_model
        self.layer = layer
        self.facet = facet
        self.device = torch.device(device)
        self.dino_extractor = self._get_dino_extractor()
        # VLAD module
        self.vlad = VLAD(num_c)
        self.clusters_fitted = False    # Flag
    
    # Extractor
    def _get_dino_extractor(self):
        return DinoV2ExtractFeatures(
            dino_model=self.dino_model, layer=self.layer,
            facet=self.facet, device=self.device)
    
    # Move the DINO model and cluster centers to another device
    def to(self, device: torch.device):
        self.device = torch.device(device)
        self.dino_extractor = self._get_dino_extractor()
        if self.clusters_fitted:
            self.vlad.c_centers = self.vlad.c_centers.to(self.device)
    
    # Wrapper around CUDA
    def cuda(self):
        self.to("cuda")
    
    # Extract image features using backbone
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        img_pt = x
        shapes = ein.parse_shape(img_pt, "b c h w")
        assert shapes["c"] == 3, "Image(s) must be RGB!"
        assert shapes["h"] % 14 == shapes["w"] % 14 == 0, \
                "Height and width should be multiple of 14 (for "\
                "patching)"
        img_pt = img_pt.to(self.device)
        # Extract features
        ret = self.dino_extractor(img_pt)   # [b, (nH*nW), dino_dim]
        return ret.to(self.device)
    
    # Get cluster centers from descriptors
    def fit(self, x: torch.Tensor) -> None:
        self.vlad.fit(x)    # x shape = (num_descs, desc_dim)
        self.clusters_fitted = True
    
    # Forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.clusters_fitted:
            raise ValueError(
                    "Cluster centers unavailable. Call 'fit'.")
        img_pt = x
        shapes = ein.parse_shape(img_pt, "b c h w")
        assert shapes["c"] == 3, "Image(s) must be RGB!"
        assert shapes["h"] % 14 == shapes["w"] % 14 == 0, \
                "Height and width should be multiple of 14 (for "\
                "patching)"
        img_pt = img_pt.to(self.device)
        # Extract features
        ret = self.dino_extractor(img_pt)   # [b, (nH*nW), dino_dim]
        gds = self.vlad.generate_multi(ret)
        return gds.to(self.device)
