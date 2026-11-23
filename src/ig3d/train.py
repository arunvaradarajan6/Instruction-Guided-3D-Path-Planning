"""Training: instruction-conditioned K-channel cost-field prediction."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from .config import DEFAULT, Config
from .costfield import denormalise
from .dataset import FieldDataset, VolumeBank
from .device import describe, get_device
from .encoder import build_encoder
from .instructions import ALL_SENTENCES, CORPUS, split_indices
from .unet3d import CostNet

from .paths import MODELS
TRAIN_SEED, VAL_SEED = 1000, 2000


def make_config(args) -> Config:
    m = dataclasses.replace(DEFAULT.model, k_channels=args.k, conditioning=args.conditioning,
                            base_channels=args.base)
    t = dataclasses.replace(DEFAULT.train, epochs=args.epochs, batch_size=args.batch,
                            lr=args.lr, seed=args.seed, train_volumes=args.train_volumes,
                            val_volumes=args.val_volumes)
    return dataclasses.replace(DEFAULT, model=m, train=t)


def load_model(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    meta = ck["meta"]
    model = CostNet(meta["emb_dim"], meta["k"], base=meta["base"], depth=meta["depth"],
                    adapter=meta.get("adapter", 0)).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    trained_on = meta.get("corpus", "v1")
    if trained_on != CORPUS:
        n_needed = max(meta["train_sent"] + meta["held_sent"]) + 1
        if n_needed > len(ALL_SENTENCES):
            raise SystemExit("{} was trained on corpus {}; rerun with IG3D_CORPUS={}".format(
                path, trained_on, trained_on))
    return model, meta


@torch.no_grad()
def predict(model, occ: np.ndarray, cond: np.ndarray, cost_cfg, dev) -> np.ndarray:
    """Occupancy (Z,Y,X) + conditioning vector (E,) -> denormalised field (K,Z,Y,X)."""
    o = torch.from_numpy(occ.astype(np.float32))[None, None].to(dev)
    e = torch.from_numpy(np.asarray(cond, np.float32))[None].to(dev)
    unit = torch.sigmoid(model(o, e).float())[0].cpu().numpy()
    return denormalise(unit, cost_cfg)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=3, choices=[1, 2, 3])
    p.add_argument("--conditioning", default="onehot", choices=["onehot", "embedding", "nli"])
    p.add_argument("--epochs", type=int, default=DEFAULT.train.epochs)
    p.add_argument("--train-volumes", type=int, default=DEFAULT.train.train_volumes)
    p.add_argument("--val-volumes", type=int, default=DEFAULT.train.val_volumes)
    p.add_argument("--batch", type=int, default=DEFAULT.train.batch_size)
    p.add_argument("--lr", type=float, default=DEFAULT.train.lr)
    p.add_argument("--base", type=int, default=DEFAULT.model.base_channels)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--adapter", type=int, default=0,
                   help="hidden width of the conditioning adapter; 0 = plain concat (IG-PRM)")
    p.add_argument("--real-cities", nargs="*", default=[],
                   help="also train on LiDAR blocks of these cities (names or 'train'); "
                        "held-out cities are refused")
    p.add_argument("--real-frac", type=float, default=0.5,
                   help="fraction of training samples drawn from the real blocks")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)

    cfg = make_config(a)
    out = Path(a.out or MODELS / "k{}_{}".format(a.k, a.conditioning))
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = get_device(a.device)
    print("device: " + describe(dev), flush=True)

    train_sent, held_sent = split_indices(seed=a.seed)
    enc = build_encoder(cfg.model, train_sent)
    print("conditioning: {} (dim {}), corpus {}: {} train / {} held-out sentences".format(
        cfg.model.conditioning, enc.dim, CORPUS, len(train_sent), len(held_sent)), flush=True)

    t0 = time.time()
    bank_tr = VolumeBank.load_or_build(cfg.train.train_volumes, TRAIN_SEED, cfg.volume,
                                       cfg.cost, "train")
    bank_va = VolumeBank.load_or_build(cfg.train.val_volumes, VAL_SEED, cfg.volume,
                                       cfg.cost, "val")
    print("volume banks ready in {:.0f}s".format(time.time() - t0), flush=True)

    ds_tr = FieldDataset(bank_tr, train_sent, enc.table, a.k, cfg.cost,
                         cfg.train.samples_per_volume, seed=a.seed)
    parts, real_cities = [ds_tr], []
    if a.real_cities:
        from .cities import HELDOUT, available, resolve
        asked = resolve(a.real_cities)
        bad = [c for c in asked if c in HELDOUT]
        if bad:
            raise SystemExit("refusing to train on held-out cities: " + ", ".join(bad))
        real_cities = available(asked)
        missing = sorted(set(asked) - set(real_cities))
        if missing:
            print("WARNING: not built, skipped: " + ", ".join(missing), flush=True)
        if not real_cities:
            raise SystemExit("none of the requested cities are built; run ig3d.cities build")
        bank_re = VolumeBank.real(real_cities, cfg.cost)
        from .instructions import ALL_LABELS, CLASSES
        from .lidar import REAL_CLASSES
        real_ids = [CLASSES.index(c) for c in REAL_CLASSES]
        real_sent = train_sent[np.isin(ALL_LABELS[train_sent], real_ids)]
        want = a.real_frac / max(1e-6, 1 - a.real_frac) * len(ds_tr)
        spv = max(1, int(np.ceil(want / len(bank_re))))
        parts.append(FieldDataset(bank_re, real_sent, enc.table, a.k, cfg.cost, spv,
                                  seed=a.seed + 7))
        print("real blocks: {} from {} (x{} per epoch, {:.0%} of samples)".format(
            len(bank_re), ", ".join(real_cities), spv,
            len(parts[1]) / (len(parts[0]) + len(parts[1]))), flush=True)
    ds_va = FieldDataset(bank_va, held_sent, enc.table, a.k, cfg.cost, 2, seed=a.seed + 99)
    pin = dev.type == "cuda"
    dl_tr = DataLoader(ConcatDataset(parts), batch_size=a.batch, shuffle=True, num_workers=0, pin_memory=pin)
    dl_va = DataLoader(ds_va, batch_size=a.batch, shuffle=False, num_workers=0, pin_memory=pin)

    model = CostNet(enc.dim, a.k, base=a.base, depth=cfg.model.depth, adapter=a.adapter)
    if a.adapter:
        seen = enc.table[train_sent]
        model.set_standardisation(seen.mean(0), seen.std(0))
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    lossf = nn.BCEWithLogitsLoss()
    amp = dev.type == "cuda"

    meta = dict(k=a.k, conditioning=cfg.model.conditioning, emb_dim=enc.dim, base=a.base,
                adapter=a.adapter, corpus=CORPUS, real_cities=real_cities,
                real_frac=a.real_frac if real_cities else 0.0,
                depth=cfg.model.depth, cond_table=enc.table.tolist(),
                train_sent=train_sent.tolist(), held_sent=held_sent.tolist(),
                config=dataclasses.asdict(cfg))
    hist, best, start = [], float("inf"), 0
    resume = out / "resume.pt"
    if resume.exists():
        ck = torch.load(resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        hist, best, start = ck["hist"], ck["best"], ck["epoch"] + 1
        torch.set_rng_state(ck["rng"].cpu())
        if dev.type == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
        print("resumed after epoch {}".format(ck["epoch"]), flush=True)
    for ep in range(start, a.epochs):
        for d in parts:
            d.set_epoch(ep)
        model.train()
        tl, nb, t0 = 0.0, 0, time.time()
        for occ, emb, tgt, _ in dl_tr:
            occ, emb, tgt = occ.to(dev), emb.to(dev), tgt.to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(occ, emb)
            loss = lossf(logits.float(), tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl += loss.item()
            nb += 1
        sched.step()

        model.eval()
        vl, vm, vn = 0.0, 0.0, 0
        with torch.no_grad():
            for occ, emb, tgt, _ in dl_va:
                occ, emb, tgt = occ.to(dev), emb.to(dev), tgt.to(dev)
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
                    logits = model(occ, emb)
                logits = logits.float()
                vl += lossf(logits, tgt).item()
                vm += (torch.sigmoid(logits) - tgt).abs().mean().item()
                vn += 1
        rec = dict(epoch=ep, train_bce=tl / nb, val_bce=vl / vn, val_mae=vm / vn,
                   secs=round(time.time() - t0, 1))
        hist.append(rec)
        print("ep {:3d}  train {:.4f}  val {:.4f}  mae {:.4f}  {}s".format(
            ep, rec["train_bce"], rec["val_bce"], rec["val_mae"], rec["secs"]), flush=True)
        if rec["val_bce"] < best:
            best = rec["val_bce"]
            torch.save(dict(model=model.state_dict(), meta=meta), out / "best.pt")
        tmp = out / "resume.pt.tmp"
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), epoch=ep, hist=hist, best=best,
                        rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if dev.type == "cuda" else None),
                   tmp)
        os.replace(tmp, resume)
    torch.save(dict(model=model.state_dict(), meta=meta), out / "last.pt")
    resume.unlink(missing_ok=True)
    (out / "history.json").write_text(json.dumps(hist, indent=2))
    print("best val bce {:.4f} -> {}".format(best, out), flush=True)


if __name__ == "__main__":
    main()
