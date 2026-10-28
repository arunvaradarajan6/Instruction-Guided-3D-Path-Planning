# Keep It Level: Direction-Dependent Cost Fields for Instruction-Guided 3D Path Planning

Code for the paper *Keep It Level: Direction-Dependent Cost Fields for Instruction-Guided 3D Path Planning* by Arun Varadarajan.

Instruction-guided planners such as IG-PRM turn a sentence ("stay low", "prefer wide gaps") into a cost map and plan on it. Their cost map stores one number per cell, so climbing three meters costs the same as sliding three meters sideways, and an instruction like **"keep it level"** cannot be expressed at all. This repository extends instruction-guided planning to 3D with **direction-dependent cost fields** that store up to three numbers per voxel:

| Channel | Meaning |
|---|---|
| `c` | cost per unit of travel |
| `a_up` | extra cost per unit of climb |
| `a_dn` | extra cost per unit of descent |

A straight move of length `L` with vertical change `Δz` costs

```
w = c̄·L + ā_up·max(Δz, 0) + ā_dn·max(−Δz, 0)
```

where each bar is the average of that channel along the move. With one channel this is exactly IG-PRM's cost lifted to 3D.

![Method overview](figures/method_overview.png)

## Main results

All numbers are regret (extra path cost relative to the optimal path under the true cost field), averaged over three training runs.

| Finding | Result |
|---|---|
| Closest single-channel field on "keep it level" | loses 14.4% (synthetic), 16.4% (Austin LiDAR) |
| Uniform split between climb and descent cost | no effect on the chosen path (max cross-regret 2.7×10⁻⁸) |
| Learned three-channel field, instruction class given | 2.1% with the exact planner, 1.9% with the PRM |
| Same model on real aerial LiDAR, no retraining | 4.1% |
| Free-form held-out sentences (NLI conditioning) | 14.5%, so language understanding is the main bottleneck |

## Repository layout

```
src/
  ig3d/
    config.py         every tunable number in one place
    volumes.py        synthetic 48³ environments (walls, windows, pillars)
    features.py       clearance, openness, and window features
    costfield.py      direction-dependent cost fields, ground truth, K-channel projections
    instructions.py   128 hand-written sentences, 8 classes, 16 each
    encoder.py        one-hot, frozen sentence embedding (MPNet + PCA), and NLI conditioning
    unet3d.py         3D U-Net cost network
    dataset.py        volume banks and the training dataset
    train.py          training one model
    planner_grid.py   exact 26-connected Dijkstra planner
    planner_prm.py    3D PRM with directed, direction-dependent edge costs
    metrics.py        regret and behavioural path metrics
    experiments.py    experiments that need no training (E0, E2, E3, sweep) and evaluation (E1)
    multiseed.py      training and evaluating every model over three seeds
    lidar.py          USGS 3DEP aerial LiDAR of downtown Austin as 48³ voxel blocks
    paperfigs.py      every figure in the paper
    viz.py            extra plots and a free-text demo
  tests/              unit tests
figures/              figures used in the paper
```

Generated data (volume caches, checkpoints, results, and the LiDAR tile) goes in `datasets/`, which is not tracked by git.

## Installation

Python 3.10 or newer. Install PyTorch first, matching your CUDA version, then the rest:

```bash
cd src
pip install torch --index-url https://download.pytorch.org/whl/cu128   # pick the build for your system
pip install -r requirements.txt
pip install "laspy[lazrs]"    # only needed for the Austin LiDAR experiments
```

The code picks CUDA, then Apple MPS, then CPU automatically.

**Language models.** The encoders use `sentence-transformers/all-mpnet-base-v2` and `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` from Hugging Face. `encoder.py` runs Hugging Face in offline mode by default, so on the first run allow downloads once:

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -m ig3d.experiments e3
```

After that, the encodings for all 128 sentences are cached in `datasets/cache/`.

Run the tests with:

```bash
pytest -m "not slow"
```

## Reproducing the paper

All commands run from `src/`.

### 1. Experiments that need no training (CPU is fine)

```bash
python -m ig3d.experiments e0      # Section IV-A: uniform climb/descent split does not change the path
python -m ig3d.experiments e2      # Fig. 3: closest one- and two-channel fields vs. the true field
python -m ig3d.experiments sweep   # Section IV-B: eight settings of the ground-truth parameters
python -m ig3d.experiments e3      # Fig. 6: linear probes on the instruction vectors
```

Results are written to `datasets/results/`.

### 2. Train and evaluate the learned models

Six models (one, two, and three channels with one-hot conditioning; three channels with the sentence embedding, NLI, and NLI plus adapter), each with three seeds:

```bash
python -m ig3d.multiseed train --workers 1    # 18 training runs
python -m ig3d.multiseed eval --workers 3     # exact planner and PRM on 30 test worlds
python -m ig3d.multiseed report               # tables, confidence intervals, paired comparisons
```

Each run takes about 19 minutes on one 8 GB laptop GPU (NVIDIA RTX PRO 2000 Blackwell). Raise `--workers` if you have more GPU memory. Training resumes automatically if interrupted. To train a single model:

```bash
python -m ig3d.train --k 3 --conditioning nli --seed 0
```

### 3. Transfer to real aerial LiDAR

Download the USGS 3DEP point-cloud tile
`USGS_LPC_TX_Central_B1_2017_stratmap17_50cm_3097433c1_LAS_2019.laz`
from the [USGS National Map](https://apps.nationalmap.gov/downloader/) and place it in `datasets/cache/lidar/`. Then:

```bash
python -m ig3d.lidar build            # voxelise into 48³ blocks (59 of 63 are used)
python -m ig3d.lidar ladder           # Fig. 3, Austin panel: no network
python -m ig3d.lidar eval             # every trained model on every block, no retraining
python -m ig3d.lidar report
```

### 4. Figures

```bash
python -m ig3d.paperfigs     # every paper figure, written to figures/
```

This needs the results from steps 1 to 3.

## Try your own instruction

```bash
python -m ig3d.viz demo --ckpt ../datasets/models/k3_nli/best.pt --text "Stay low and keep it level."
```

This predicts a cost field from your sentence in a synthetic test world, plans on it with the PRM, and saves a figure of the path next to the shortest path (`--index` picks a different world). It needs an `nli` or `embedding` checkpoint, since one-hot models cannot read free text, and the language model must be available locally (see Installation).

## Citation

```bibtex
@misc{varadarajan2026keepitlevel,
  title  = {Keep It Level: Direction-Dependent Cost Fields for Instruction-Guided 3D Path Planning},
  author = {Varadarajan, Arun},
  year   = {2026},
  note   = {arXiv preprint}
}
```

## Acknowledgments

This work builds on IG-PRM (Bao and Yonetani, ICRA 2025). The LiDAR data is public-domain data from the USGS 3D Elevation Program.

## License

See [LICENSE](LICENSE).
