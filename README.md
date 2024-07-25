# AnyLoc on Torch Hub

This repository is for AnyLoc's release on Torch Hub.

Please see: [anyloc.github.io](https://anyloc.github.io/) or the main [AnyLoc repository](https://github.com/AnyLoc/AnyLoc) for the actual work.

> **Note**: This is a work-in-progress (not ready for public release yet), mainly for internal use. Please don't use it if you're not in the AnyLoc team. Use `force_reload = True` while the API is unstable.

Please open issues about this work in [AnyLoc/AnyLoc](https://github.com/AnyLoc/AnyLoc) with label `torch.hub`

## Table of contents

- [AnyLoc on Torch Hub](#anyloc-on-torch-hub)
    - [Table of contents](#table-of-contents)
    - [Tutorial](#tutorial)
    - [TODO](#todo)
    - [References](#references)

## Tutorial

Basic usage

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
# Help about individual functions
r = torch.hub.help("AnyLoc/DINO", "get_vlad_model")
print(r)
```

## TODO

- [x] AnyLoc-VLAD-DINOv2
- [ ] AnyLoc-VLAD-DINO
- [ ] AnyLoc-VLAD-DINOv2-PCA
- [ ] AnyLoc-VLAD-DINO-PCA

## References

- [torch.hub](https://pytorch.org/docs/stable/hub.html)
    - [Download directory](https://pytorch.org/docs/stable/hub.html#where-are-my-downloaded-models-saved)
