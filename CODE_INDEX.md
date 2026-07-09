# Code Index

This repository pairs a static student website with local prototype code.

## Website

- `website/index.html` - static interactive learning site
- `website/styles.css` - site styles
- `website/app.js` - browser interactions, quizzes, simulations, and local-backend detection
- `website/assets/` - small public figures and reference assets used by the website

## Module Code

- `module1_wfm/` - WFM image-to-image learning prototype
- `module2_ecg/` - ECG beat classification prototype
- `module3_fall/` - public fall/motion sequence modeling prototype
- `module4_resistive_array/` - resistive array inverse-modeling prototype
- `backend_server.py` - local-only teaching backend for realtime demos

## Not Included in GitHub

The repository intentionally excludes large or local-only files through `.gitignore`, including:

- virtual environments
- large datasets
- PyTorch checkpoints
- generated experiment outputs
- local run folders

The GitHub Pages site should be treated as the public static version. Realtime training remains a local instructor/developer mode.
