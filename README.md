# Interactive Machine Learning Educational Platform

Student-facing educational website for learning machine learning through real biomedical case studies.

The public-facing design is based on interactive saved experiments. A local
backend can optionally run realtime PyTorch training for instructor demos,
development, and generating new saved results.

The project currently contains two guided modules:

- Module 1: WFM image-to-image learning with a pix2pix-style conditional GAN workflow.
- Module 2: ECG time-series classification with a 1D CNN workflow using MIT-BIH beat segments.
- Module 3: Motion/fall-risk sequence modeling and Holomotion-style feature interpretation.
- Module 4: Resistive array inverse modeling for local sensor-map reconstruction.

## Current Prototype

The website prototype is in:

```text
website/
```

To preview locally:

```powershell
.\start_website.ps1
```

Then open:

```text
http://127.0.0.1:4173/
```

To publish the static student website with GitHub Pages, see:

```text
DEPLOY_GITHUB_PAGES.md
```

## Learning Design

Each module is designed as a guided lab rather than a static report.

Student workflow:

1. Choose a module.
2. Learn the model concept.
3. Inspect the dataset.
4. Choose training options.
5. Run a guided training workflow.
6. Test the model.
7. Reflect on the result and limitations.

## Repository Structure

```text
.
|-- backend_server.py
|-- module1_wfm/
|-- module2_ecg/
|-- module3_fall/
|-- module4_resistive_array/
|-- website/
|   |-- index.html
|   |-- styles.css
|   |-- app.js
|   `-- assets/
|-- Module2_ECG_Student_Facing_Documentation.md
|-- Module2_ECG_Student_Facing_Documentation.docx
|-- week3_plan.md
|-- week3_ecg_results.md
`-- README.md
```

## Public Website vs Local Realtime Mode

The website has two intended modes:

| Mode | Audience | What works |
|---|---|---|
| Public website | Students online | Lessons, dataset explorers, saved checkpoints, saved metrics, reflection prompts |
| Local realtime mode | Developer / instructor / advanced lab | Real PyTorch training through the local backend |

Browsers cannot run PyTorch training directly. Realtime training requires
`backend_server.py`, local datasets, Python dependencies, and the local project
paths configured in the backend.

## Realtime ECG Training

The local backend exposes:

```text
/api/ecg/train
```

The website detects this endpoint automatically. When the backend is running,
the ECG Train page shows an advanced local training option. When the backend is
not running, students can still explore saved experiment results online.

Current local assumptions:

- ECG scripts live in `C:\Users\10131\PycharmProjects\PythonProject5\module2_ecg`.
- MIT-BIH prepared data lives in `C:\Users\10131\PycharmProjects\PythonProject5\data\ecg`.
- Training outputs are written to `runs/`, which is ignored by Git.

## Realtime WFM Training

The local backend also exposes:

```text
/api/wfm/train
```

The WFM module can start a real pix2pix-style GAN training run from the website
when the local backend is running. Because GAN training is slower than ECG
classification, the public student flow should primarily use saved checkpoints.
Local WFM training is an advanced preview mode for demonstrations and generating
new saved result images.

## Data and Model Files

Large datasets, trained checkpoints, virtual environments, and full experiment outputs should not be committed directly to GitHub.

Recommended external storage:

- MIT-BIH ECG data: download or store separately.
- WFM raw image dataset: store separately.
- PyTorch checkpoints: store separately or use Git LFS if needed.
- Large papers and references: store in a shared folder unless they are small and redistributable.

## Current Status

- Module 1 student-facing website flow: in progress.
- Module 2 student-facing website flow: in progress.
- References page: includes WFM source paper context.
- Realtime ECG training backend: first local prototype.
- Realtime WFM training backend: first local prototype.

## Next Development Steps

- Generalize realtime backend paths and configuration.
- Add WFM image pair splitter interaction.
- Add ECG class imbalance simulator.
- Add student prediction prompts before metric reveal.
