# GitHub Pages Deployment

This repository is ready to publish the static student website with GitHub Pages.

## Recommended Setup

1. Create a GitHub repository, for example `ml-learning-lab`.
2. Upload this project to the repository.
3. Edit `website/deploy-config.js` and replace:

```js
window.SITE_REPO_URL = "https://github.com/YOUR_USERNAME/ml-learning-lab";
```

with the real repository URL.

4. In GitHub, open the repository settings:

```text
Settings -> Pages -> Build and deployment -> Source -> GitHub Actions
```

5. Push to the `main` branch. The workflow in `.github/workflows/pages.yml` will publish the `website/` folder.

The public site URL will usually look like:

```text
https://YOUR_USERNAME.github.io/ml-learning-lab/
```

## What Gets Published

Only the `website/` folder is published as the public GitHub Pages site.

The Python module code remains in the GitHub repository and is linked from the website's References page. Large datasets, trained checkpoints, virtual environments, and experiment outputs should stay out of GitHub.

## Current Code Links

- Module 3 fall/motion prototype: `module3_fall/`
- Module 4 resistive array prototype: `module4_resistive_array/`
- Local backend prototype: `backend_server.py`

Module 1 and Module 2 realtime training currently depend on local project paths and datasets. For public deployment, the website uses saved figures and guided interactions instead of running training online.
