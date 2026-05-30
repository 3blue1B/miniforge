# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

`miniforge` is a personal PyTorch / deep-learning learning playground, not a single
cohesive application. It contains a series of independent experiments plus one
deployable web app. There is no package, no test suite, and no shared module — each
script is meant to be run on its own.

## The pieces (and how they relate)

- **`app.py`** — The only production/deployable artifact. A Gradio app that reads the
  first 3 pages of an uploaded PDF, summarizes each page through the **DeepSeek** chat
  API (via the `openai` SDK pointed at `https://api.deepseek.com/v1`), and synthesizes
  the summary to an MP3 with `gTTS`. Requires the `DEEPSEEK_API_KEY` env var. Designed
  for Render deployment: it binds `0.0.0.0` and reads the `PORT` env var (default 7860).
  This app is what `requirements.txt` (`gradio`, `openai`, `PyPDF2`, `gTTS`) covers.

- **`week1/`** — Standalone learning scripts, each runnable independently:
  - `try.py` — minimal `nn.Linear` linear-regression demo on synthetic data.
  - `lea1.py` — feed-forward `Model` (4→8→8→3) trained on the Iris dataset (pulled from
    a remote gist URL). Saves weights to `../iris_model.pth`.
  - `lea2.py` — reloads `iris_model.pth` for inference. **The `Model` class is copy-pasted
    here from `lea1.py`** — the two definitions must stay architecturally identical or
    `load_state_dict` will fail.
  - `lea3.py` — `ConvolutionalNetwork` CNN trained on MNIST (auto-downloaded to
    `week1/cnn_data/`). Tracks best test loss and saves the best checkpoint to `mnist_cnn.pth`.

- **`templates/index.html`** — A self-contained canvas-based MNIST digit-recognition UI
  that `POST`s the drawn image to a `/predict` endpoint. **No Python backend serving this
  endpoint exists in the repo yet** — it is a front-end ahead of its server.

- **`learn0.ipynb`** — Colab scratchpad notebook (links back to this repo on Colab).

- **`*.pth`** — Committed trained weights (`iris_model.pth`, `mnist_cnn.pth`, and a copy
  in `week1/`).

## Critical convention: scripts assume their own directory as CWD

The Iris scripts use the relative path `../iris_model.pth`, so they expect to be run
**from inside `week1/`** (writing/reading the checkpoint at the repo root). `lea3.py`
uses CWD-relative paths for both `./cnn_data` and `mnist_cnn.pth`. Always `cd` into the
script's directory before running, and be aware that the same model name (`mnist_cnn.pth`)
exists both at the repo root and in `week1/`.

## Running things

```bash
# The deployable PDF-summarizer app
export DEEPSEEK_API_KEY=...      # required, or summaries error out
pip install -r requirements.txt
python app.py                    # serves on http://0.0.0.0:7860

# Learning scripts (run from their own directory)
cd week1
python lea1.py                   # trains Iris model, writes ../iris_model.pth
python lea2.py                   # loads ../iris_model.pth (needs lea1 run first)
python lea3.py                   # trains MNIST CNN, downloads data, saves mnist_cnn.pth
python try.py
```

The `week1` scripts depend on `torch`, `torchvision`, `pandas`, `scikit-learn`, and
`matplotlib`, which are **not** in `requirements.txt` (that file is scoped to `app.py`).
Install them separately when working in `week1/`.

## Notes for making changes

- There are no tests, linters, or build steps configured. The GitHub Actions workflow
  (`.github/workflows/blank.yml`) is the default placeholder that only echoes "Hello,
  world!" — CI does not validate the code.
- If you change the `Model` architecture in `week1/lea1.py`, mirror the exact change in
  `week1/lea2.py`, otherwise the saved `iris_model.pth` will no longer load.
- If you implement the `/predict` backend for `templates/index.html`, it expects a PNG
  data URL and should serve `templates/index.html` (Flask-style `render_template` is
  already imported in `lea2.py` as a starting point).
