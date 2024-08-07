# AnyLoc on Torch Hub

This repository is for AnyLoc's release on Torch Hub.

Please see: [anyloc.github.io](https://anyloc.github.io/) or the main [AnyLoc repository](https://github.com/AnyLoc/AnyLoc) for the actual work.

> **Note**: This is a work-in-progress (not ready for public release yet), mainly for internal use. Please don't use it if you're not in the AnyLoc team. Use `force_reload = True` while the API is unstable.

Please open issues about this work in [AnyLoc/AnyLoc](https://github.com/AnyLoc/AnyLoc) with label `torch.hub`

## Table of contents

- [AnyLoc on Torch Hub](#anyloc-on-torch-hub)
    - [Table of contents](#table-of-contents)
    - [Tutorial](#tutorial)
        - [Map-specific Vocabulary](#map-specific-vocabulary)
    - [TODO](#todo)
    - [References](#references)

## Tutorial

Install the following

```bash
pip install einops      # Codebase uses this
pip install fast_pytorch_kmeans     # For VLAD codebase
```

Basic usage (pass in the `domain`, `backbone`, and set the device to `cuda` for
using a GPU)

```py
import torch
model = torch.hub.load("AnyLoc/DINO", "get_vlad_model", 
        domain="indoor", backbone="DINOv2", device="cuda")
# Images
img = torch.rand(1, 3, 224, 224)
# Result: VLAD descriptors of shape [1, 49152]
res = model(img)
```

It also supports batching

```py
# Images
img = torch.rand(16, 3, 224, 224)
# Result: VLAD descriptors of shape [16, 49152]
res = model(img)
```

You can get more help from

```py
# List of functions
print(torch.hub.list("AnyLoc/DINO"))
# Help about an individual function - like "get_vlad_model"
r = torch.hub.help("AnyLoc/DINO", "get_vlad_model")
print(r)
```

### Map-specific Vocabulary

This is to use your own dataset for calculating the VLAD clusters

```py
import einops as ein
# Load model
model = torch.hub.load("AnyLoc/DINO", "get_vlad_model", 
        domain=None, backbone="DINOv2", device="cuda")
# Extract features
imgs = torch.rand(16, 3, 224, 224)  # Database images
res = model.extract(imgs)
res_all = ein.rearrange(res, "B N D -> (B N) D")
# Fit VLAD (to get cluster centers)
model.fit(res_all)
# Get the descriptors
img = torch.rand(1, 3, 224, 224)    # Inference images
gd = model(img) # Global descriptors
```

## TODO

- [x] AnyLoc-VLAD-DINOv2
- [ ] AnyLoc-VLAD-DINO
- [ ] AnyLoc-VLAD-DINOv2-PCA
- [ ] AnyLoc-VLAD-DINO-PCA

## References

- [torch.hub](https://pytorch.org/docs/stable/hub.html)
    - [Download directory](https://pytorch.org/docs/stable/hub.html#where-are-my-downloaded-models-saved)
