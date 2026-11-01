"""Deep learning models of the remaining hold time (PyTorch).

Survival models predict a discrete-time hazard of release on the bins of tabprep.EDGES
(logistic-hazard likelihood, censoring-aware; remaining times beyond 168 h are censored at
168 h). The survival curve gives both the mean remaining time and the expected gain
E[(min(R, 168) - delta)^+] of Equation (2) for any delta, including hold-specific deltas.

Encoders
  mlp    : categorical embeddings + MLP
  ft     : FT-Transformer (Gorishniy et al. 2021): one token per feature + CLS token
  lotseq : mlp encoder of the current hold + Transformer over the lot's previous holds, with
           the current hold as query token

Decision-focused ranker: the same encoder with a scalar head, trained with a listwise
(ListNet) loss on review epochs so that the score orders the open holds of one review by
their realised gain; it can be warm-started from a trained survival encoder.
"""
from __future__ import annotations
import copy, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from .tabprep import EDGES, group_index  # noqa: F401

K_BINS = len(EDGES) - 1


def device_auto(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _emb_dim(card: int) -> int:
    return int(min(32, max(4, round(1.6 * card ** 0.56))))


# --------------------------------------------------------------------------------------- encoders
class MLPEncoder(nn.Module):
    def __init__(self, n_num, cards, hidden=256, layers=2, dropout=0.1):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, _emb_dim(c), padding_idx=0) for c in cards])
        d_in = n_num + sum(_emb_dim(c) for c in cards)
        blocks, d = [], d_in
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        self.net = nn.Sequential(*blocks)
        self.out_dim = hidden

    def forward(self, b):
        z = [b["num"]] + [e(b["cat"][:, j]) for j, e in enumerate(self.embs)]
        return self.net(torch.cat(z, dim=1))


class FTEncoder(nn.Module):
    def __init__(self, n_num, cards, d=64, layers=3, heads=4, dropout=0.1, ffn_mult=2):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_num, d) / math.sqrt(d))
        self.b = nn.Parameter(torch.zeros(n_num, d))
        self.cat_emb = nn.ModuleList([nn.Embedding(c, d, padding_idx=0) for c in cards])
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, d * ffn_mult, dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.tr = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out_dim = d

    def forward(self, b):
        x = b["num"].unsqueeze(-1) * self.w + self.b
        c = torch.stack([e(b["cat"][:, j]) for j, e in enumerate(self.cat_emb)], dim=1)
        tok = torch.cat([self.cls.expand(x.shape[0], -1, -1), x, c], dim=1)
        return self.norm(self.tr(tok)[:, 0])


class LotSeqEncoder(nn.Module):
    """Current hold (MLP) attends over the lot's previous holds (Transformer)."""

    def __init__(self, n_num, cards, seq_cards, n_seq_num, hidden=256, layers=2, dropout=0.1, d_seq=64,
                 seq_layers=2, heads=4):
        super().__init__()
        self.tab = MLPEncoder(n_num, cards, hidden, layers, dropout)
        self.ev_cat = nn.ModuleList([nn.Embedding(c, d_seq, padding_idx=0) for c in seq_cards])
        self.ev_num = nn.Linear(n_seq_num, d_seq)
        self.pos = nn.Parameter(torch.zeros(1, 64, d_seq))
        self.q = nn.Linear(hidden, d_seq)
        layer = nn.TransformerEncoderLayer(d_seq, heads, d_seq * 2, dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.tr = nn.TransformerEncoder(layer, seq_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_seq)
        self.out_dim = hidden + d_seq

    def forward(self, b):
        u = self.tab(b)
        K = b["seq_num"].shape[1]
        assert K <= self.pos.shape[1], "K must be <= 64"
        ev = self.ev_num(b["seq_num"]) + sum(e(b["seq_cat"][:, :, j]) for j, e in enumerate(self.ev_cat))
        ev = ev + self.pos[:, :K]
        tok = torch.cat([self.q(u).unsqueeze(1), ev], dim=1)
        pad = torch.cat([torch.zeros_like(b["mask"][:, :1]), b["mask"]], dim=1)
        s = self.norm(self.tr(tok, src_key_padding_mask=pad)[:, 0])
        return torch.cat([u, s], dim=1)


def make_encoder(arch: str, meta: dict, hp: dict) -> nn.Module:
    if arch == "mlp":
        return MLPEncoder(meta["n_num"], meta["cards"], hp["hidden"], hp["layers"], hp["dropout"])
    if arch == "ft":
        return FTEncoder(meta["n_num"], meta["cards"], hp["d"], hp["layers"], hp["heads"], hp["dropout"])
    if arch == "lotseq":
        return LotSeqEncoder(meta["n_num"], meta["cards"], meta["seq_cards"], meta["n_seq_num"], hp["hidden"],
                             hp["layers"], hp["dropout"], hp["d_seq"], hp["seq_layers"], hp.get("heads", 4))
    raise ValueError(arch)


class Net(nn.Module):
    def __init__(self, arch, meta, hp, out_dim):
        super().__init__()
        self.arch = arch
        self.trunk = make_encoder(arch, meta, hp)
        self.head = nn.Linear(self.trunk.out_dim, out_dim)

    def forward(self, b):
        return self.head(self.trunk(b))


# --------------------------------------------------------------------------------------- data
KEYS = ("num", "cat", "seq_cat", "seq_num", "mask")


def _batch(D: dict, idx: np.ndarray, dev, arch: str) -> dict:
    keys = KEYS if arch == "lotseq" else ("num", "cat")
    return {k: torch.as_tensor(D[k][idx]).to(dev, non_blocking=True) for k in keys}


def surv_labels(R: np.ndarray, event: np.ndarray):
    """Bin of the observed remaining time and event indicator (censored beyond 168 h)."""
    y = np.clip(np.searchsorted(EDGES, R, "right") - 1, 0, K_BINS - 1)
    ev = (event.astype(bool) & (R < EDGES[-1])).astype(np.float32)
    return y.astype(np.int64), ev


def surv_loss(logits, y, ev):
    k = torch.arange(logits.shape[1], device=logits.device)[None, :]
    mask = (k <= y[:, None]).float()
    target = ((k == y[:, None]).float() * ev[:, None])
    l = Fn.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (l * mask).sum(1).mean()


def listnet_loss(s, gid, w):
    """Listwise softmax cross-entropy per review group; groups with zero total gain are skipped."""
    G = int(gid.max().item()) + 1
    mx = torch.full((G,), -1e30, device=s.device).scatter_reduce(0, gid, s.detach(), "amax", include_self=True)
    z = torch.exp(s - mx[gid])
    lse = torch.log(torch.zeros(G, device=s.device).index_add(0, gid, z)) + mx
    logp = s - lse[gid]
    tot = torch.zeros(G, device=s.device).index_add(0, gid, w)
    t = w / tot[gid].clamp_min(1e-12)
    lg = -torch.zeros(G, device=s.device).index_add(0, gid, t * logp)
    valid = tot > 0
    return lg[valid].mean() if valid.any() else (s * 0).sum()


def _seed_all(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------------------- survival
def train_surv(arch, meta, hp, Dtr, ytr, Dva, yva, seed=1, dev=None, max_epochs=40, patience=5, batch=1024, log=print):
    dev = dev or device_auto()
    _seed_all(seed)
    model = Net(arch, meta, hp, K_BINS).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(1, patience // 2))
    rng = np.random.default_rng(seed)
    (y_tr, e_tr), (y_va, e_va) = ytr, yva
    best, best_state, wait, hist = np.inf, None, 0, []
    n = len(y_tr)
    for ep in range(max_epochs):
        t0 = time.time(); model.train(); tot = 0.0
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            b = _batch(Dtr, idx, dev, arch)
            loss = surv_loss(model(b), torch.as_tensor(y_tr[idx], device=dev), torch.as_tensor(e_tr[idx], device=dev))
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += loss.item() * len(idx)
        va = eval_surv_loss(model, arch, Dva, y_va, e_va, dev, batch * 4)
        sched.step(va)
        hist.append(dict(epoch=ep, train=tot / n, val=va, sec=time.time() - t0))
        log(f"      [{arch}] epoch {ep:2d} train {tot / n:.4f} val {va:.4f} ({time.time() - t0:.0f}s)")
        if va < best - 1e-4:
            best, best_state, wait = va, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, dict(best_val=best, epochs=len(hist), history=hist)


@torch.no_grad()
def eval_surv_loss(model, arch, D, y, e, dev, batch=4096):
    model.eval(); tot = 0.0
    for i in range(0, len(y), batch):
        idx = np.arange(i, min(i + batch, len(y)))
        b = _batch(D, idx, dev, arch)
        tot += surv_loss(model(b), torch.as_tensor(y[idx], device=dev), torch.as_tensor(e[idx], device=dev)).item() * len(idx)
    return tot / max(len(y), 1)


@torch.no_grad()
def predict_surv(model, arch, D, dev=None, batch=4096) -> np.ndarray:
    """Survival curve at EDGES, shape (n, len(EDGES))."""
    dev = dev or next(model.parameters()).device
    model.eval(); out = []
    n = len(D["num"])
    for i in range(0, n, batch):
        idx = np.arange(i, min(i + batch, n))
        hz = torch.sigmoid(model(_batch(D, idx, dev, arch))).float().cpu().double()  # float64 on CPU (MPS has none)
        S = torch.cumprod(1 - hz, dim=1)
        out.append(torch.cat([torch.ones(len(idx), 1, dtype=S.dtype), S], dim=1).numpy())
    return np.concatenate(out) if out else np.zeros((0, len(EDGES)))


# --------------------------------------------------------------------------------------- ranker
def train_ranker(arch, meta, hp, Dtr, gtr, wtr, Dva, gva, wva, seed=1, dev=None, max_epochs=30, patience=4,
                 groups_per_batch=32, init_trunk: dict | None = None, log=print):
    """gtr = (starts, sizes) of review groups; wtr = per-row target weight (gain ** power)."""
    dev = dev or device_auto()
    _seed_all(seed)
    model = Net(arch, meta, hp, 1).to(dev)
    if init_trunk is not None:
        model.trunk.load_state_dict(init_trunk)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    rng = np.random.default_rng(seed)
    best, best_state, wait, hist = np.inf, None, 0, []
    for ep in range(max_epochs):
        t0 = time.time(); model.train(); tot, nb = 0.0, 0
        order = rng.permutation(len(gtr[0]))
        for i in range(0, len(order), groups_per_batch):
            gsel = order[i:i + groups_per_batch]
            idx, gid = _group_rows(gtr, gsel)
            s = model(_batch(Dtr, idx, dev, arch)).squeeze(1)
            loss = listnet_loss(s, torch.as_tensor(gid, device=dev), torch.as_tensor(wtr[idx], device=dev, dtype=s.dtype))
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += loss.item(); nb += 1
        va = eval_ranker_loss(model, arch, Dva, gva, wva, dev, groups_per_batch * 4)
        hist.append(dict(epoch=ep, train=tot / max(nb, 1), val=va, sec=time.time() - t0))
        log(f"      [rank-{arch}] epoch {ep:2d} train {tot / max(nb, 1):.4f} val {va:.4f} ({time.time() - t0:.0f}s)")
        if va < best - 1e-4:
            best, best_state, wait = va, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, dict(best_val=best, epochs=len(hist), history=hist)


def _group_rows(g, gsel):
    starts, sizes = g
    idx = np.concatenate([np.arange(starts[j], starts[j] + sizes[j]) for j in gsel])
    gid = np.repeat(np.arange(len(gsel)), sizes[gsel])
    return idx, gid


@torch.no_grad()
def eval_ranker_loss(model, arch, D, g, w, dev, gpb=128):
    model.eval(); tot, nb = 0.0, 0
    for i in range(0, len(g[0]), gpb):
        gsel = np.arange(i, min(i + gpb, len(g[0])))
        idx, gid = _group_rows(g, gsel)
        s = model(_batch(D, idx, dev, arch)).squeeze(1)
        tot += listnet_loss(s, torch.as_tensor(gid, device=dev), torch.as_tensor(w[idx], device=dev, dtype=s.dtype)).item()
        nb += 1
    return tot / max(nb, 1)


@torch.no_grad()
def predict_score(model, arch, D, dev=None, batch=4096) -> np.ndarray:
    dev = dev or next(model.parameters()).device
    model.eval(); out = []
    n = len(D["num"])
    for i in range(0, n, batch):
        idx = np.arange(i, min(i + batch, n))
        out.append(model(_batch(D, idx, dev, arch)).squeeze(1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
