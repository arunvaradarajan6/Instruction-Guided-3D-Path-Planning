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
    instructions.py   8 classes: the paper's 128 sentences (v1), 256 extra training phrasings (v2),
                      and an 80-sentence hard test set that is never trained on
    encoder.py        one-hot, frozen sentence embedding (MPNet + PCA), and NLI conditioning
    unet3d.py         3D U-Net cost network
    dataset.py        volume banks and the training dataset
    train.py          training one model
    planner_grid.py   exact 26-connected Dijkstra planner
    planner_prm.py    3D PRM with directed, direction-dependent edge costs
    metrics.py        regret and behavioural path metrics
    experiments.py    experiments that need no training (E0, E2, E3, sweep) and evaluation (E1)
    multiseed.py      training and evaluating every model over three seeds
    lidar.py          USGS 3DEP aerial LiDAR as 48³ voxel blocks (Austin, and any city below)
    cities.py         more LiDAR (3DEP, AWS copy): 11 training cities, Seattle and Miami held out with Austin
    sim/              planned path -> flyable trajectory -> Gazebo + PX4 SITL flight -> flown regret
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
pip install "laspy[lazrs]" pyproj    # only needed for the LiDAR experiments
```

The code picks CUDA, then Apple MPS, then CPU automatically.

**Language models.** The encoders use `sentence-transformers/all-mpnet-base-v2` and `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` from Hugging Face. `encoder.py` runs Hugging Face in offline mode by default, so on the first run allow downloads once:

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -m ig3d.experiments e3
```

After that, the encodings for every sentence are cached in `datasets/cache/`. Set `IG3D_CORPUS=v2` to train and evaluate with the extra phrasings; the default `v1` reproduces the paper, and v2 checkpoints and results go to `datasets/models_v2/` and `datasets/results_v2/`.

Run the tests with:

```bash
pytest -m "not slow"
```


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
