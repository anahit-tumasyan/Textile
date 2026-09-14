"""
train.py  (vision branch)
=========================

Training / evaluation loop for ``FibreBlendNet``, written for the deployment
target actually in play: **a single Kaggle GPU** (T4 16GB, P100 16GB, or an L4/
A100 when the queue is kind), 12-hour session ceiling, ~30 GPU-hours a week.

What that constraint changed, concretely:

* **Mixed precision, dtype chosen from the card.** bf16 where the hardware has it
  (no loss scaler, no overflow babysitting); fp16 + ``GradScaler`` on T4/P100,
  which lack bf16. The Newton-Schulz square root is forced back to fp32 inside
  ``SecondOrderPool`` regardless — that is the one op in this graph that fp16
  genuinely breaks.
* **Checkpoint every epoch, resume from anywhere.** A 12-hour ceiling means every
  serious run is a resumed run. A training script that cannot resume is unusable
  here, not merely inconvenient.
* **Bags are the batch.** Effective batch = ``batch_size * n_patches`` images
  through the backbone. batch=4, P=8 is 32 patches of 384px per step and fits a
  T4 comfortably; use ``--grad-checkpoint`` and raise P before raising batch,
  because the covariance estimate improves with more patches per bag while a
  larger bag batch only smooths gradients.
* **channels_last** memory format: free throughput on tensor-core cards.

Smoke run, no data required (~2 min on CPU):

    python -m src.vision.train --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (
    FabricSwatch,
    SwatchBagDataset,
    compute_lds_weights,
    mass_to_area,
    area_to_mass,
)
from .calibration import empirical_coverage, fit_temperature
from .losses import BlendCompositionLoss
from .model import FibreBlendNet, composition_from_alpha, uncertainty_from_alpha

__all__ = ["pick_amp_dtype", "ModelEMA", "train_one_epoch", "evaluate", "main"]


# --------------------------------------------------------------------------- #
# Precision / schedule utilities
# --------------------------------------------------------------------------- #
def pick_amp_dtype(device: torch.device) -> tuple[torch.dtype | None, bool]:
    """
    Returns (autocast dtype, whether a GradScaler is needed).

    Note on the API: ``torch.cuda.amp.autocast`` is deprecated since torch 2.4 in
    favour of ``torch.amp.autocast(device_type=...)``. The new spelling is used
    throughout; it is the same mechanism, and it is the one that still works on
    current Kaggle images.
    """
    if device.type != "cuda":
        return None, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, False        # A100 / L4 / newer
    return torch.float16, True              # T4 / P100


def cosine_warmup(step: int, total: int, warmup: int, min_ratio: float = 0.02) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


class ModelEMA:
    """
    Exponential moving average of weights.

    Worth its ~2 lines on a corpus of a few thousand swatches: the last-epoch
    weights of a small-data run are noisy, and the EMA is reliably 0.1-0.3 pp
    better on MAE for no extra compute.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.steps = 0
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        # Warmup-corrected decay. A flat 0.999 has a ~1000-step time constant, so
        # on a short run (or the first epoch of any run) the shadow is still
        # mostly the random initialisation and the EMA metric is meaningless.
        self.steps += 1
        d = min(self.decay, (1.0 + self.steps) / (10.0 + self.steps))
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:
                s.copy_(v)

    def copy_to(self, model: torch.nn.Module) -> None:
        model.load_state_dict({k: v for k, v in self.shadow.items()}, strict=False)

    @contextmanager
    def applied(self, model: torch.nn.Module):
        """
        Temporarily swap the EMA weights into ``model`` for evaluation, then put
        the live weights back.

        Without this the EMA is decorative: it accumulates every step, gets
        checkpointed, and is never actually the thing being scored or selected on.
        """
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        try:
            self.copy_to(model)
            yield model
        finally:
            model.load_state_dict(backup, strict=True)


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    scaler=None,
    amp_dtype=None,
    ema: ModelEMA | None = None,
    sched=None,
    max_norm: float = 1.0,
    log_every: int = 20,
) -> dict:
    model.train()
    loss_fn.train()
    running: dict[str, float] = {}
    n = 0
    t0 = time.time()

    for i, batch in enumerate(loader):
        patches = batch["patches"].to(device, non_blocking=True)
        target = batch["mass"].to(device, non_blocking=True)
        weight = batch["weight"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # --- the AMP step ---------------------------------------------------
        with torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
        ):
            out = model(patches)
            # The loss is computed OUTSIDE autocast's reduced precision by
            # casting inside BlendCompositionLoss: lgamma and log of a near-zero
            # composition are not fp16-safe, and a silent inf here shows up much
            # later as an unexplained NaN.
            loss, stats = loss_fn(out, target, weight)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if sched is not None:
            sched.step()
        if ema is not None:
            ema.update(model)

        for k, v in stats.items():
            running[k] = running.get(k, 0.0) + v
        n += 1
        if log_every and i % log_every == 0:
            print(f"  step {i:5d}  loss {stats['loss']:.4f}  mae {stats['mae_pp']:.2f} pp")

    out = {k: v / max(n, 1) for k, v in running.items()}
    out["epoch_sec"] = time.time() - t0
    return out


@torch.no_grad()
def collect_alphas(model, loader, device, amp_dtype=None):
    """Run the model over a loader and return (alpha, target) as numpy arrays."""
    model.eval()
    alphas, targets = [], []
    for batch in loader:
        patches = batch["patches"].to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
        ):
            out = model(patches)
        alphas.append(out["alpha"].float().cpu())
        targets.append(batch["mass"])
    return torch.cat(alphas).numpy(), torch.cat(targets).numpy()


def evaluate(model, loader, device, amp_dtype=None, alpha_scale: float = 1.0) -> dict:
    """
    Reports the metrics this project is actually judged on, in percentage points,
    so the vision branch is directly comparable with the NIR table in the README.

    ``coverage_90`` is the fraction of swatches whose true cotton fraction falls
    inside the model's own 90% credible interval. If the model is calibrated this
    is ~0.90; well below means the Dirichlet is overconfident and its uncertainty
    cannot be used to gate a sorting decision — which is most of its value.

    ``alpha_scale`` is the post-hoc temperature from ``calibration.fit_temperature``.
    It cannot change mae/rmse/r2 — the Dirichlet mean is invariant to it — so if
    those move when you pass it, something else is wrong.
    """
    from .calibration import empirical_coverage as _cov

    a, t = collect_alphas(model, loader, device, amp_dtype)
    a = a / alpha_scale
    p = a / a.sum(axis=1, keepdims=True)

    err = (p[:, 0] - t[:, 0]) * 100.0
    ss_res = float((err ** 2).sum())
    ss_tot = float((((t[:, 0] - t[:, 0].mean()) * 100.0) ** 2).sum()) + 1e-9

    # HDI, not the equal-tailed interval: a central interval provably cannot
    # cover a vertex label (y = 1, 0), and pure swatches are a real, common class.
    covered = _cov(a, t, level=0.90, kind="hdi")

    return {
        "mae_pp": float(np.abs(err).mean()),
        "rmse_pp": float(np.sqrt((err ** 2).mean())),
        "r2": 1.0 - ss_res / ss_tot,
        "coverage_90": covered,
        "mean_uncertainty": float((a.shape[1] / a.sum(1)).mean()),
        "n": int(len(p)),
    }


# --------------------------------------------------------------------------- #
# Corpora
# --------------------------------------------------------------------------- #
def synthetic_corpus(n: int, size: int, seed: int = 0) -> list[FabricSwatch]:
    """
    A bimodal label distribution on purpose — real blend corpora pile up at
    100/0, 65/35 and 50/50, and a uniform toy corpus would quietly hide the
    imbalance machinery this pipeline exists to exercise.
    """
    from .fabric_simulator import render_swatch

    rng = np.random.default_rng(seed)
    modes = np.array([1.00, 0.65, 0.50, 0.35, 0.0])
    out = []
    for i in range(n):
        m = modes[rng.integers(0, len(modes))] + rng.normal(0, 0.04)
        cotton_mass = float(np.clip(m, 0.0, 1.0))
        mass = np.array([cotton_mass, 1.0 - cotton_mass])
        area = mass_to_area(mass)
        img = render_swatch(area[0], size=size, rng=rng)
        out.append(FabricSwatch(swatch_id=f"sim{i:04d}", mass=mass, array=img))
    return out


def split_by_swatch(swatches, frac=0.2, seed=0):
    """Split by swatch id. Never split by patch — see SwatchBagDataset docstring."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(swatches))
    k = int(round(len(swatches) * frac))
    val = [swatches[i] for i in idx[:k]]
    train = [swatches[i] for i in idx[k:]]
    return train, val


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="FibreBlend vision training")
    ap.add_argument("--smoke", action="store_true", help="synthetic run, no data needed")
    ap.add_argument("--smoke-n", type=int, default=16, help="number of synthetic swatches")
    ap.add_argument("--smoke-size", type=int, default=192, help="synthetic swatch side in px")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--patch", type=int, default=384)
    ap.add_argument("--n-patches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head-lr-mult", type=float, default=10.0)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--mix-prob", type=float, default=0.5)
    # Calibration knobs. If coverage_90 comes back well below 0.90 the Dirichlet
    # is overconfident and its uncertainty cannot gate a sorting decision --
    # raise --l-dir, and check --nll-warmup is not a large fraction of the run.
    ap.add_argument("--l-dir", type=float, default=1.0, help="Dirichlet NLL weight")
    ap.add_argument("--l-mae", type=float, default=2.0, help="L1 weight")
    ap.add_argument("--nll-warmup", type=int, default=-1,
                    help="steps to ramp the NLL term; -1 = 10%% of total steps")
    ap.add_argument("--no-notch", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--freeze-stages", type=int, default=2)
    ap.add_argument("--out", type=str, default="results/vision")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--calib-frac", type=float, default=0.2,
                    help="fraction of TRAIN held out to fit the Dirichlet temperature")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, needs_scaler = pick_amp_dtype(device)
    print(f"device={device}  amp={amp_dtype}  scaler={needs_scaler}")

    if args.smoke:
        args.patch = min(args.patch, args.smoke_size // 2)
        args.workers = 0
        swatches = synthetic_corpus(n=args.smoke_n, size=args.smoke_size, seed=args.seed)
    else:
        raise SystemExit(
            "No real corpus is wired up yet. Build a list[FabricSwatch] from your "
            "swatch images + reference compositions and pass it here; --smoke runs "
            "the full pipeline on the simulator in the meantime."
        )

    # Three-way split. The calibration set is carved out of TRAIN, not val:
    # fitting the temperature on the same swatches you report coverage on is
    # circular and reports ~0.90 however broken the model is.
    train_sw, val_sw = split_by_swatch(swatches, frac=0.25, seed=args.seed)
    train_sw, calib_sw = split_by_swatch(train_sw, frac=args.calib_frac, seed=args.seed + 1)
    w = compute_lds_weights(np.stack([s.mass for s in train_sw]))
    print(f"train={len(train_sw)} calib={len(calib_sw)} val={len(val_sw)}  "
          f"lds weight range [{w.min():.2f}, {w.max():.2f}]")

    common = dict(patch=args.patch, n_patches=args.n_patches, notch=not args.no_notch)
    ds_tr = SwatchBagDataset(train_sw, train=True, mix_prob=args.mix_prob,
                             weights=w, seed=args.seed, **common)
    ds_va = SwatchBagDataset(val_sw, train=False, mix_prob=0.0, **common)
    ds_ca = SwatchBagDataset(calib_sw, train=False, mix_prob=0.0, **common)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers, pin_memory=(device.type == "cuda"),
                       drop_last=len(ds_tr) > args.batch_size)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.workers, pin_memory=(device.type == "cuda"))
    dl_ca = DataLoader(ds_ca, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.workers, pin_memory=(device.type == "cuda"))

    model = FibreBlendNet(
        n_fibres=2,
        pretrained=not args.no_pretrained,
        freeze_stages=args.freeze_stages,
        grad_checkpoint=args.grad_checkpoint,
    ).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {n_par/1e6:.1f}M total, {n_tr/1e6:.1f}M trainable")

    # Discriminative LRs: the pretrained backbone wants a gentle rate, the
    # randomly-initialised neck and head want an order of magnitude more.
    backbone, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone if name.startswith(("stage_early", "stage_last")) else head).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": args.lr},
         {"params": head, "lr": args.lr * args.head_lr_mult}],
        weight_decay=args.weight_decay,
    )
    total_steps = max(len(dl_tr) * args.epochs, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: cosine_warmup(s, total_steps, warmup=min(200, total_steps // 10))
    )
    scaler = torch.amp.GradScaler(device.type, enabled=needs_scaler)
    # A fixed 500-step NLL warmup is longer than a short run, which leaves the
    # Dirichlet term permanently ramped-down and the model overconfident. Scale
    # it to the run instead.
    warmup = args.nll_warmup if args.nll_warmup >= 0 else max(total_steps // 10, 1)
    loss_fn = BlendCompositionLoss(
        l_dir=args.l_dir, l_mae=args.l_mae, nll_warmup=warmup
    ).to(device)
    ema = ModelEMA(model)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, best = 0, float("inf")
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["sched"])
        ema.shadow = {k: v.to(device) for k, v in ck["ema"].items()}
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    history = []
    for epoch in range(start_epoch, args.epochs):
        print(f"epoch {epoch + 1}/{args.epochs}")
        tr = train_one_epoch(model, dl_tr, optimizer, loss_fn, device,
                             scaler=scaler if needs_scaler else None,
                             amp_dtype=amp_dtype, ema=ema, sched=sched)
        raw = evaluate(model, dl_va, device, amp_dtype=amp_dtype)
        with ema.applied(model):
            va = evaluate(model, dl_va, device, amp_dtype=amp_dtype)
        print(f"  train {tr}\n  val(raw) {raw}\n  val(ema) {va}")
        history.append({"epoch": epoch, "train": tr, "val": va, "val_raw": raw})

        # Model selection is on the EMA weights, because those are the weights
        # that ship. Selecting on raw and shipping EMA (or vice versa) is a
        # quiet way to make the reported number not describe the model.
        improved = va["mae_pp"] < best
        best = min(best, va["mae_pp"])
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "sched": sched.state_dict(), "ema": ema.shadow, "epoch": epoch,
             "best": best, "args": vars(args)},
            out_dir / "last.pt",
        )
        if improved:
            torch.save({"model": ema.shadow, "args": vars(args)}, out_dir / "best.pt")

    # --- post-hoc calibration ---------------------------------------------
    # Fitted on the held-out calibration split, then reported on val. The point
    # prediction is mathematically invariant to this scaling, so mae/rmse/r2 must
    # come back identical; only coverage_90 moves.
    with ema.applied(model):
        ca_alpha, ca_y = collect_alphas(model, dl_ca, device, amp_dtype)
        temp, cal_info = fit_temperature(ca_alpha, ca_y, level=0.90)
        va_cal = evaluate(model, dl_va, device, amp_dtype=amp_dtype, alpha_scale=temp)
    print(f"calibration: s={temp:.3f}  {cal_info}")
    if not cal_info["attainable"]:
        # Print the note rather than a fixed message: `attainable` is False for two
        # different reasons (unreachable coverage vs too few calibration swatches)
        # and stating the wrong one sends the reader after the wrong problem.
        print(f"  WARNING: uncalibrated -- {cal_info['note']}")
    print(f"  val(ema, calibrated) {va_cal}")

    ck = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    ck["alpha_scale"] = temp
    ck["calibration"] = cal_info
    torch.save(ck, out_dir / "best.pt")

    (out_dir / "history.json").write_text(
        json.dumps({"history": history, "alpha_scale": temp,
                    "calibration": cal_info, "val_calibrated": va_cal}, indent=2)
    )
    print(f"done. best val MAE {best:.2f} pp -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
