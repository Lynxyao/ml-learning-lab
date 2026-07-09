# Module 1 WFM Code

This folder contains the lightweight PyTorch source code for the WFM image-to-image learning prototype.

## Files

- `train_torch.py` - train the pix2pix-style WFM model
- `test_torch.py` - generate held-out test visualizations
- `network_torch.py` - generator and discriminator definitions
- `util.py` - data loading and image utilities
- `requirements.txt` - CPU/local dependency notes
- `requirements_gpu.txt` - optional GPU dependency notes

## Public Website Use

The GitHub Pages website uses saved result images for the public student flow. Full realtime training is intended for local instructor/developer use because it depends on local datasets and compute.
