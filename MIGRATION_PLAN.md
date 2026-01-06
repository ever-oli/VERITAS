# Migration Plan: Clean Slate & Repo Testing

The previous CodeSpace ran out of disk space. This repository reset is intentional.

## Objective
The goal is to test two specific repositories (`audio2score` and `music2score`) in a clean environment, using `uv` for dependency management.

## Instructions for the Next Agent

1.  **Environment Check**:
    -   Ensure `uv` is installed (`pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`).
    -   Ensure `ffmpeg` is installed (required by both pipelines).
    -   Check python versions (Project requires >= 3.10, < 3.12).

2.  **Clone Repositories**:
    -   Clone `https://github.com/ac1965/audio2score.git`
    -   Clone `https://github.com/ac1965/music2score.git`

3.  **Tangle Source Code**:
    -   Both repositories use `README.org` files as the source of truth (Literate Programming).
    -   **CRITICAL**: You MUST "tangle" (extract) the code from `README.org` into the actual python files before running them. The cloned repos might have outdated `src/` folders.
    -   Use a python script to parse the `#+begin_src ... :tangle path/to/file` blocks in `README.org` and write them to disk.

4.  **Install & Test `audio2score`**:
    -   `cd audio2score`
    -   `uv venv --python 3.11`
    -   `uv pip install -e .` (Note: `pyproject.toml` might depend on older torch versions, you may need to relax constraints to `torch>=2.1`, `torchaudio>=2.1` if 2.7 fails).
    -   Test: Run `python -m audio2score.cli --help` or pipeline entry.

5.  **Install & Test `music2score`**:
    -   `cd music2score`
    -   `uv venv --python 3.11`
    -   `uv pip install -r env/common/requirements.txt`
    -   Test: Run provided scripts in `env/common/scripts/`.

## Context from Previous Session
-   **Disk Space**: The main blocker was disk space. Be mindful of large torch downloads (pip cache, uv cache).
-   **Dependencies**: The projects are strict about Python versions (3.11 recommended).
-   **Structure**: The user prefers `uv` over Conda.
