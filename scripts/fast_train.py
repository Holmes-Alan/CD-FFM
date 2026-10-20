"""
fast_train.py
==============
Single-GPU PyTorch Lightning trainer for the fast pipeline. All data lives on
the GPU (fast_data.py), so a step is ~model compute only (Flower ~45 ms vs
~6 s/step with the old DataLoader). Batches are drawn on the GPU inside
training_step; the DataLoader Lightning requires is a dummy that yields
placeholders (no cost). One process per GPU (configs/commands/05_train.txt). Every experiment axis is a flag:

  --backbone fno|flower         encoder
  --objective fm|regression     flow matching, or the deterministic MSE baseline
                                (same network; velocity net used as a direct
                                predictor at x=0, t=0)
  --noise field|iid             correlated RFF noise field vs i.i.d. per point
  --noise-ls                    noise-field length scale (deg)
  --noise-features              number of random Fourier features in the noise field
  --noise-kernel rbf|powerlaw   RFF spectral density; powerlaw = spectrum-matched S(k)~k^-beta
  --noise-beta, --noise-lmax-km, --noise-lmin-km   powerlaw slope and wavelength band (km)
  --lifting-dim                 encoder width (capacity-matched backbone comparison)
  --mc-weight                   soft mass-conservation loss weight (0 = off)
  --doy-cond                    day-of-year (cos,sin) into the velocity net
  --holdout lapland             never sample that region during training

Validation every --val-every steps on a fixed set (12 dates x 20k pixels of
the val year), with EMA weights and ensemble sampling: RMSE (single member and
ensemble mean), MAE, CRPS, spread -- physical degC, per variable, separately
for held-out pixels when --holdout is set. Logged to TensorBoard and to
runs/<name>/metrics.jsonl, so parallel runs can be compared by learning curve.
"""
import argparse
import copy
import json
import math
import os
import time
import warnings
from datetime import timedelta

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import LightningEnvironment
from torch.utils.data import DataLoader, IterableDataset

from fast_data import (STATIC_DIMS, TARGET_VARS, GPUData, doy_features, mu_sigma, noise_field, noise_kwargs,
                       ode_sample, pick_device, to_phys)
from model import DownscalingFlowModel, TwoTimeVelocityNet
from paths import CDFFM_ROOT  # noqa: E402  (set CDFFM_ROOT)

RUNS_ROOT = CDFFM_ROOT + "/runs"
VARS = ["TG", "TX", "TN"]
# The DataLoader is a placeholder (batches are sampled on the GPU), so
# Lightning's "use more workers" advice doesn't apply.
warnings.filterwarnings("ignore", ".*does not have many workers.*")


class _Placeholder(IterableDataset):
    """Lightning needs a DataLoader; the real batch is sampled on the GPU in
    training_step, so this just yields a placeholder forever (train) or
    once (val)."""

    def __init__(self, n=None):
        self.n = n

    def __iter__(self):
        i = 0
        while self.n is None or i < self.n:
            yield torch.zeros(1)
            i += 1


def build_model(args, era5_lat, era5_lon):
    m = _build_model(args, era5_lat, era5_lon)
    if args.objective == "flowmap":  # two-time net; weights come from the teacher / checkpoint
        m.velocity_net = TwoTimeVelocityNet.from_teacher(m.velocity_net)
    return m


def load_teacher(name, device, era5_lat, era5_lon):
    """Frozen EMA weights of a trained flow-matching run at its best validation checkpoint."""
    from fast_eval import load_run  # lazy: fast_eval imports this module
    targs, teacher, step = load_run(name, device, era5_lat, era5_lon, "best")
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"[flowmap] teacher {name} (step {step})", flush=True)
    return targs, teacher


def _build_model(args, era5_lat, era5_lon):
    return DownscalingFlowModel(
        era5_in_ch=4, lat_grid=era5_lat, lon_grid=era5_lon, backbone=args.backbone,
        n_out=len(TARGET_VARS[getattr(args, "target", "temp")]), extra_cond_dim=2 if args.doy_cond else 0,
        static_dim=STATIC_DIMS[getattr(args, "static", "none")],
        encoder_kwargs={"lifting_dim": args.lifting_dim} if getattr(args, "lifting_dim", None) else None)


def condition(model, era5, lat, lon, doy, doy_cond, static=None):
    """static: (B,N,S) per-point static covariates (or None) -> into the LIIF query MLP."""
    cond = model.condition(model.encode(era5), lat, lon, static)
    if doy_cond:
        cond = torch.cat([cond, doy_features(doy)[:, None, :].expand(-1, lat.shape[1], -1)], dim=-1)
    return cond


def lr_factor(step, args):
    """Linear warmup, then cosine decay to 10% of the base LR at max_steps."""
    if step < args.warmup:
        return (step + 1) / args.warmup
    p = min(1.0, (step - args.warmup) / max(1, args.max_steps - args.warmup))
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))


def flowmap_loss(model, teacher, b, cond, args):
    """Flow-map distillation: u(x_t, t, s) regresses the teacher's average velocity
    (x_s - x_t) / (s - t), with x_s from the teacher ODE (RK4, args.fm_substeps steps) started at the
    interpolant x_t; on a fraction of points s = t and the target is the teacher's velocity itself."""
    x1, valid = b["x1"], b["valid"].float()
    B, N, C = x1.shape
    if args.noise == "field":
        z0 = noise_field(b["lat"], b["lon"], C, **noise_kwargs(args))
    else:
        z0 = torch.randn_like(x1)
    t = torch.rand(B, N, device=x1.device)
    diag = torch.rand(B, N, device=x1.device) < args.fm_diag_prob
    s = torch.where(diag, t, t + (1 - t) * torch.rand(B, N, device=x1.device))
    xt = (1 - t)[..., None] * z0 + t[..., None] * x1
    with torch.no_grad():
        v0 = teacher.velocity(xt, cond, t)
        h = (s - t) / args.fm_substeps
        x, tk = xt, t
        for _ in range(args.fm_substeps):
            k1 = teacher.velocity(x, cond, tk)
            k2 = teacher.velocity(x + 0.5 * h[..., None] * k1, cond, tk + 0.5 * h)
            k3 = teacher.velocity(x + 0.5 * h[..., None] * k2, cond, tk + 0.5 * h)
            k4 = teacher.velocity(x + h[..., None] * k3, cond, tk + h)
            x = x + h[..., None] / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            tk = tk + h
        avg = (x - xt) / (s - t).clamp(min=1e-6)[..., None]
        target = torch.where(diag[..., None], v0, avg)
    u = model.velocity(xt, cond, t, s)
    return (((u - target) ** 2) * valid).sum() / valid.sum().clamp(min=1)


def compute_losses(model, data, args, teacher=None):
    b = data.sample_batch(args.batch_size, args.n_scattered, args.n_clusters, args.points_per_cluster)
    B, N, _ = b["x1"].shape
    cond = condition(model, b["era5"], b["lat"], b["lon"], b["doy"], args.doy_cond, b["static"])
    x1, valid = b["x1"], b["valid"].float()

    if args.objective == "flowmap":
        return flowmap_loss(model, teacher, b, cond, args), torch.zeros((), device=x1.device)
    if args.objective == "fm":
        if args.noise == "field":
            z0 = noise_field(b["lat"], b["lon"], x1.shape[-1], **noise_kwargs(args))
        else:
            z0 = torch.randn_like(x1)
        t = torch.rand(B, N, device=x1.device)
        xt = (1 - t)[..., None] * z0 + t[..., None] * x1
        v = model.velocity(xt, cond, t)
        main = (((v - (x1 - z0)) ** 2) * valid).sum() / valid.sum().clamp(min=1)
        x1_hat = xt + (1 - t)[..., None] * v
    else:
        pred = model.velocity(torch.zeros_like(x1), cond, torch.zeros(B, N, device=x1.device))
        main = (((pred - x1) ** 2) * valid).sum() / valid.sum().clamp(min=1)
        x1_hat = pred

    mc = torch.zeros((), device=x1.device)
    if args.mc_weight > 0:
        # Physical-space soft mass conservation on TG: each sampled ERA5 cell's
        # mean predicted temperature vs. the ERA5 value, scaled by the local
        # NGCD sigma so it is O(1) like the main loss. Cluster points occupy a
        # fixed (cells x points) block, so this is a plain mean, no loop.
        ns, C, K = args.n_scattered, args.n_clusters, args.points_per_cluster
        cm, cv = data.gather_coeffs(b["idx"][:, ns:])  # (B,C*K,3,5)
        mu, sig = mu_sigma(cm[:, :, 0:1], cv[:, :, 0:1], b["doy"], data.sigma_floor)  # TG only, keep the channel axis
        mu, sig = mu[..., 0], sig[..., 0]  # (B,C*K)
        tg_phys = (x1_hat[:, ns:, 0] * sig + mu).reshape(B, C, K)
        scale = sig.reshape(B, C, K).mean(-1)
        mc = (((tg_phys.mean(-1) - b["era5_cell_phys"]) / scale) ** 2).mean()
    return main, mc


@torch.no_grad()
def validate(model, data, args):
    P = data.val_pix.shape[0]
    lat, lon = data.lat[data.val_pix], data.lon[data.val_pix]
    C = data.n_out
    held = data.holdout_mask[data.val_pix]
    groups = {"all": torch.ones_like(held)}
    if args.holdout:
        groups["holdout"], groups["inside"] = held, ~held
    acc = {g: {k: torch.zeros(C, device=lat.device) for k in ("se1", "seM", "ae", "crps", "spread", "n")}
           for g in groups}
    M = 1 if args.objective == "regression" else args.val_members
    for d in range(len(data.val_dates)):
        doy = data.val_doy[d:d + 1]
        cond = condition(model, data.val_era5[d:d + 1], lat[None], lon[None], doy, args.doy_cond,
                         data.static[data.val_pix][None] if data.static is not None else None)
        if args.objective == "regression":
            ens = model.velocity(torch.zeros(1, P, C, device=lat.device), cond, torch.zeros(1, P, device=lat.device))
        else:
            ens = ode_sample(model, cond.expand(M, -1, -1), lat[None].expand(M, -1), lon[None].expand(M, -1),
                             args.val_ode_steps, noise=args.noise, noise_kw=noise_kwargs(args),
                             flowmap=args.objective == "flowmap")
        ens_phys = to_phys(data, ens, data.val_pix, doy)  # (M,P,C) degC or mm/day
        truth = to_phys(data, data.val_targets[d][None], data.val_pix, doy)  # (1,P,C)
        ok = torch.isfinite(truth[0])
        tr = torch.nan_to_num(truth[0])
        e1 = ens_phys[0] - tr
        eM = ens_phys.mean(0) - tr
        crps = (ens_phys - tr[None]).abs().mean(0)
        if M > 1:
            crps = crps - 0.5 * (ens_phys[:, None] - ens_phys[None]).abs().mean((0, 1))
        spread = ens_phys.std(0, unbiased=False) if M > 1 else torch.zeros_like(tr)
        for g, gm in groups.items():
            m = (ok & gm[:, None]).float()
            a = acc[g]
            a["se1"] += (e1 ** 2 * m).sum(0)
            a["seM"] += (eM ** 2 * m).sum(0)
            a["ae"] += (eM.abs() * m).sum(0)
            a["crps"] += (crps * m).sum(0)
            a["spread"] += (spread * m).sum(0)
            a["n"] += m.sum(0)
    out = {}
    for g, a in acc.items():
        n = a["n"].clamp(min=1)
        for i, v in enumerate(data.vars):
            out[f"{g}/{v}/rmse_1member"] = float(torch.sqrt(a["se1"][i] / n[i]))
            out[f"{g}/{v}/rmse_ensmean"] = float(torch.sqrt(a["seM"][i] / n[i]))
            out[f"{g}/{v}/mae"] = float(a["ae"][i] / n[i])
            out[f"{g}/{v}/crps"] = float(a["crps"][i] / n[i])
            out[f"{g}/{v}/spread"] = float(a["spread"][i] / n[i])
    return out


class FastFFM(pl.LightningModule):
    def __init__(self, args, data):
        super().__init__()
        self.args = args
        self.data = data  # plain object (not an nn.Module): its GPU tensors are not checkpointed
        self.model = build_model(args, data.era5_lat, data.era5_lon)
        self.teacher = None
        if args.objective == "flowmap":
            targs, teacher = load_teacher(args.teacher, data.device, data.era5_lat, data.era5_lon)
            for k in ("backbone", "lifting_dim", "doy_cond", "static", "target", "noise", "noise_ls", "noise_features",
                      "noise_kernel", "noise_beta", "noise_lmin_km", "noise_lmax_km"):
                # runs trained before an option existed have no entry for it -> the option's default
                defaults = dict(target="temp", static="none", noise_kernel="rbf", noise_beta=2.4,
                                noise_lmin_km=2.0, noise_lmax_km=500.0, noise_features=64, lifting_dim=None)
                tv = getattr(targs, k, defaults.get(k))
                if tv != getattr(args, k, None):
                    raise SystemExit(f"--{k.replace('_', '-')} differs from teacher {args.teacher}: "
                                     f"{getattr(args, k, None)} vs {tv}")
            sd = {k: v for k, v in teacher.state_dict().items() if not k.startswith("velocity_net.")}
            missing = self.model.load_state_dict(sd, strict=False)
            assert all(k.startswith("velocity_net.") for k in missing.missing_keys), missing
            self.model.velocity_net = TwoTimeVelocityNet.from_teacher(teacher.velocity_net)
            for n_, p in self.model.named_parameters():  # encoder + LIIF query frozen: same cond as the teacher
                p.requires_grad_(n_.startswith("velocity_net."))
            self.teacher = [teacher.to(data.device).eval()]  # list: keep it out of Lightning's state_dict
        self.ema = copy.deepcopy(self.model)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.run_dir = f"{RUNS_ROOT}/{args.name}"
        self.t0 = time.time()
        self._t_last = None

    def training_step(self, batch, batch_idx):
        main, mc = compute_losses(self.model, self.data, self.args, self.teacher[0] if self.teacher else None)
        self.log("train/loss_main", main, on_step=True, prog_bar=True)
        self.log("train/loss_mc", mc, on_step=True)
        return main + self.args.mc_weight * mc

    def on_train_batch_end(self, outputs, batch, batch_idx):
        with torch.no_grad():
            d = self.args.ema
            for pe, pm in zip(self.ema.parameters(), self.model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        step = self.global_step
        every = 200
        if step % every == 0:
            now = time.time()
            if self._t_last is not None:
                ms = (now - self._t_last) / every * 1000
                self.log("train/ms_per_step", ms)
                if step % 1000 == 0 or step <= 1000:
                    print(f"[step {step}] {ms:.0f} ms/step (incl. any validation in the window), "
                          f"elapsed {(now - self.t0) / 60:.1f} min", flush=True)
            self._t_last = now

    def validation_step(self, batch, batch_idx):
        self._run_val()

    def on_train_end(self):
        if self.global_step % self.args.val_every != 0:
            self._run_val(log_to_lightning=False)

    def _run_val(self, log_to_lightning=True):
        res = validate(self.ema.eval(), self.data, self.args)
        if log_to_lightning:
            self.log_dict({f"val/{k}": v for k, v in res.items()})
        res.update(step=int(self.global_step), elapsed_min=(time.time() - self.t0) / 60)
        with open(f"{self.run_dir}/metrics.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")
        print(f"[val step {res['step']}] " + " ".join(
            f"{v}: rmse1={res[f'all/{v}/rmse_1member']:.2f} rmseM={res[f'all/{v}/rmse_ensmean']:.2f} "
            f"crps={res[f'all/{v}/crps']:.2f} spread={res[f'all/{v}/spread']:.2f}" for v in self.data.vars), flush=True)
        if self.args.holdout:
            print("            holdout " + " ".join(
                f"{v}: crps={res[f'holdout/{v}/crps']:.2f} (inside {res[f'inside/{v}/crps']:.2f})"
                for v in self.data.vars), flush=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.args.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, self.args))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def train_dataloader(self):
        return DataLoader(_Placeholder(), batch_size=None)

    def val_dataloader(self):
        return DataLoader(_Placeholder(1), batch_size=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--backbone", choices=["fno", "flower"], default="flower")
    ap.add_argument("--objective", choices=["fm", "regression", "flowmap"], default="fm",
                    help="flowmap: distil a few-step flow map from --teacher (encoder frozen)")
    ap.add_argument("--teacher", default=None, help="flowmap: trained fm run to distil")
    ap.add_argument("--fm-substeps", type=int, default=8, help="flowmap: teacher RK4 steps per (t, s) target")
    ap.add_argument("--fm-diag-prob", type=float, default=0.25, help="flowmap: fraction of points with s = t")
    ap.add_argument("--noise", choices=["field", "iid"], default="field")
    ap.add_argument("--noise-ls", type=float, default=0.5)
    ap.add_argument("--noise-features", type=int, default=64,
                    help="random Fourier features in the noise field; more = each sample closer to a true GP")
    ap.add_argument("--noise-kernel", choices=["rbf", "powerlaw"], default="rbf")
    ap.add_argument("--noise-beta", type=float, default=2.4,
                    help="powerlaw kernel: per-mode spectral slope (NGCD residual spectrum ~2.4)")
    ap.add_argument("--noise-lmax-km", type=float, default=500.0, help="powerlaw kernel: longest wavelength")
    ap.add_argument("--noise-lmin-km", type=float, default=2.0, help="powerlaw kernel: shortest wavelength (grid: 2 km)")
    ap.add_argument("--lifting-dim", type=int, default=None,
                    help="encoder width (channels at the top U-Net level; default 64). FNO: 16 -> ~7.7M params, "
                         "8 -> ~1.9M (Flower at 64 is 3.2M). Must be divisible by 8 (GroupNorm/heads).")
    ap.add_argument("--mc-weight", type=float, default=0.1)
    ap.add_argument("--doy-cond", action="store_true")
    ap.add_argument("--target", choices=["temp", "precip"], default="temp",
                    help="temp = TG/TX/TN (z-scores); precip = NGCD RR (p_dry + bulk table + GPD tail -> Gaussian)")
    ap.add_argument("--static", choices=["none", "clim", "clim_rr"], default="none",
                    help="per-pixel static covariates into the LIIF query; clim = 17 features from the "
                         "NGCD 1981-2015 climatology (terrain proxy: sub-grid mean offset, diurnal range, "
                         "seasonal amplitude, winter/summer variability)")
    ap.add_argument("--holdout", choices=["lapland"], default=None)
    ap.add_argument("--train-years", type=str, default="2011,2012,2013,2014,2015")
    ap.add_argument("--val-year", type=int, default=2017)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-scattered", type=int, default=2048)
    ap.add_argument("--n-clusters", type=int, default=16)
    ap.add_argument("--points-per-cluster", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=60000)
    ap.add_argument("--time-budget-min", type=float, default=None)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-members", type=int, default=8)
    ap.add_argument("--val-ode-steps", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from runs/<name>/ckpt/last.ckpt if present")
    ap.add_argument("--device", default="auto", help="'auto' (first GPU, else CPU), 'cuda:0', 'cpu', ...")
    ap.add_argument("--data-store", choices=["auto", "device", "cpu", "disk"], default="auto",
                    help="where the training targets live (~44 GB for 15 years): on the GPU, in host RAM, or "
                         "memory-mapped from disk; 'auto' picks the first that fits. Batches are identical.")
    args = ap.parse_args()

    pl.seed_everything(args.seed, workers=False)
    device = pick_device(args.device)
    print(f"[device] {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""),
          flush=True)
    run_dir = f"{RUNS_ROOT}/{args.name}"
    os.makedirs(run_dir, exist_ok=True)
    with open(f"{run_dir}/args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    data = GPUData([int(y) for y in args.train_years.split(",")], args.val_year, device,
                   holdout=args.holdout, seed=0,  # fixed val set: comparable across training seeds
                   static=args.static, target=args.target, store=args.data_store)
    if args.target == "precip" and args.mc_weight > 0:
        raise SystemExit("--mc-weight applies to TG only; use --mc-weight 0 with --target precip")
    module = FastFFM(args, data)
    print(f"[model] {args.backbone} objective={args.objective} "
          f"params={sum(p.numel() for p in module.model.parameters()):,}", flush=True)

    ckpt_cb = ModelCheckpoint(dirpath=f"{run_dir}/ckpt", every_n_train_steps=args.ckpt_every,
                              save_top_k=-1, save_last=True, filename="step{step:06d}")
    trainer = pl.Trainer(
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=[device.index or 0] if device.type == "cuda" else 1,
        # Plain single-process environment (also under Slurm: don't read its variables as a distributed setup).
        plugins=[LightningEnvironment()],
        max_steps=args.max_steps,
        max_time=timedelta(minutes=args.time_budget_min) if args.time_budget_min else None,
        val_check_interval=args.val_every, check_val_every_n_epoch=None,
        limit_val_batches=1, num_sanity_val_steps=0,
        gradient_clip_val=args.grad_clip,
        logger=TensorBoardLogger(save_dir=run_dir, name="tb"),
        log_every_n_steps=100,
        callbacks=[ckpt_cb],
        enable_progress_bar=False,
    )
    last = f"{run_dir}/ckpt/last.ckpt"
    trainer.fit(module, ckpt_path=last if (args.resume and os.path.exists(last)) else None)
    print(f"[done] {trainer.global_step} steps in {(time.time() - module.t0) / 60:.1f} min -> {run_dir}",
          flush=True)


if __name__ == "__main__":
    main()
