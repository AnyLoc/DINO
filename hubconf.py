# Configuration (and loading) file for Torch Hub

dependencies = [
    'torch', 'torchvision',     # Basic PyTorch
    'einops',       # Functionalities (utilities)
    'fast_pytorch_kmeans', # KMeans clustering (in VLAD)
]

# %%
import os
import torch
from torch import nn
import model_wrapper as mw
from typing import Optional


# %%
# Released cluster centers (for direct use/without VLAD training)
AVAIL_VLADCC_FILES = [    # Available VLAD Cluster Center files
    # DINO Model - ViT - Layer - Facet - Num CCenters - Domain
    "dinov2_vitg14_l31_value_c32_indoor_c_centers.pt",
    "dinov2_vitg14_l31_value_c32_urban_c_centers.pt",
    "dinov2_vitg14_l31_value_c32_aerial_c_centers.pt",
    "dinov2_vitg14_l31_value_c32_structured_c_centers.pt",
    "dinov2_vitg14_l31_value_c32_unstructured_c_centers.pt",
    "dinov2_vitg14_l31_value_c32_global_c_centers.pt",
]
# Base URL for the releases (on GitHub)
BASER_URL = "https://github.com/AnyLoc/DINO/releases/download/v1"


# %%
def get_vlad_model(domain: Optional[str], backbone: str = "DINOv2", 
            vit_model: str = "ViT-G/14", vit_layer: int = 31, 
            vit_facet: str = "Value", num_c: int = 32, 
            device: torch.device = "cpu") \
            -> nn.Module:
    """
        Load an AnyLoc-VLAD-[backbone] model from torch.hub
        The default settings are for AnyLoc-VLAD-DINOv2; and the
        'indoor' domain is used. The domain would depend on the 
        deployment setting/use case (environment).
        
        Parameters:
        - domain (str):     Domain for cluster centers. Should be
                            "indoor", "urban", "aerial", "structured",
                            "unstructured", or "global" (if string);
                            or None (for using non-preloaded cluster 
                            centers).
        - backbone (str):   The backbone to use. Should be "DINOv2" or
                            "DINOv1".
        - vit_model (str):  The ViT model (architecture) to use. Must
                            be compatible with the backbone. "/" and
                            "-" are ignored.
        - vit_layer (int):  The layer to use for feature extraction.
        - vit_facet (str):  The ViT facet to use for extraction.
        - num_c (int):      Number of cluster centers to use (for
                            VLAD clustering).
        - device (torch.device):    Device for model; "cpu" or "cuda"
        
        Notes:
        - All string arguments are converted to lower case.
    """
    # Parse arguments (assert types)
    backbone = str(backbone).lower()
    vit_model = str(vit_model)\
        .replace("/", "").replace("-", "").lower()
    vit_layer = int(vit_layer)
    vit_facet = str(vit_facet).lower()
    num_c = int(num_c)
    if type(domain) == str: # Using pre-loaded cluster centers
        domain = str(domain).lower()
        assert domain in ["indoor", "urban", "aerial", "structured",
                "unstructured", "global"], f"Invalid {domain = }"
        # Make sure cluster centers are available
        cc_fname = f"{backbone}_{vit_model}_l{vit_layer}_"\
                f"{vit_facet}_c{num_c}_{domain}_c_centers.pt"
        if cc_fname not in AVAIL_VLADCC_FILES:
            raise ValueError(
                    f"Cluster centers for {cc_fname} not found!")
        # Download the cluster centers
        _ex = lambda x: os.path.realpath(os.path.expanduser(x))
        loc_pthub_path = _ex(torch.hub.get_dir())
        loc_path = f"{loc_pthub_path}/checkpoints/anyloc_files"
        if os.path.isdir(loc_path) == False:    # Create if not there
            os.makedirs(loc_path)
        print(f"Storing (torch.hub) cache in: {loc_path}")
        cc_path = f"{loc_path}/{cc_fname}"  # Path for cluster centers
        if os.path.isfile(cc_path):
            print(f"Cluster centers already downloaded: {cc_path}")
        else:
            torch.hub.download_url_to_file(f"{BASER_URL}/{cc_fname}", 
                    cc_path)
        c_centers = torch.load(cc_path)
        assert c_centers.shape[0] == num_c, \
                "Cluster centers corrupted"
    
    # Return model
    model = nn.Identity()   # Placeholder
    device = torch.device(device)
    if type(domain) == str and backbone == "dinov2":    # Preloaded
        model = mw.AnyLocVladDinov2(c_centers, 
                dino_model=f"{backbone}_{vit_model}", layer=vit_layer,
                facet=vit_facet, num_c=num_c, device=device)
        print(f"Model with {domain = } vocabulary loaded")
    elif domain is None and backbone == "dinov2":
        model = mw.AnyLocVladNoCacheDinov2(
                dino_model=f"{backbone}_{vit_model}", layer=vit_layer,
                facet=vit_facet, num_c=num_c, device=device)
        print("Loaded model with no cluster centers")
    else:
        raise NotImplementedError(f"{backbone = } not implemented!")
    return model


