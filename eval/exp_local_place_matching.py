"""NEGATIVE RESULT -- local matching does not resolve the borderline band.
Kept so nobody spends another evening on it. Run: exp_local_place_frames.py first.

                    band 3v33   session24 v band33   easy 6v6
    global             0.5051            0.9533       1.0000
    patch              0.7626            0.7607       0.5833
    patch_nobody       0.7626            0.7336       0.5000
    sift               0.4040            0.6622       0.5417
    sift_nobody        0.4545            0.7077       0.6528

    production env_raw on band 3v33 = 0.8182

Read the columns together and patch matching is not a candidate: its 0.7626
on the band rests on THREE positives, and the same scorer manages 0.5833 on
the easy pairs the global axis separates perfectly, and loses the
well-powered session-vs-band comparison 0.7607 to 0.9533. A method that
cannot do the easy case is not going to fix the hard one. SIFT+RANSAC is
below chance on the band.

WHY, from the match rates -- fraction of candidate matches surviving RANSAC:

    patch   same_session 0.00429   band same 0.00429   band different 0.00386
    sift    same_session 0.09500   band same 0.06667   band different 0.07333

Positives and hard negatives are indistinguishable, and the absolute numbers
are the noise floor: ~0.4% of patches surviving means RANSAC fitting an
affine to four or five coincidental matches, not recurring structure. The one
place real correspondence shows up is same-session SIFT (max 1.14, i.e. >100
inliers) -- two segments of ONE recording, where the viewpoints genuinely
overlap.

That is the physical answer. Classical place recognition assumes overlapping
views of rigid structure. Two visits to the same bench on different days, on
a head-mounted camera, share almost no viewpoint: lighting differs, clutter
differs, the head is always moving, and 8 sampled frames per clip make the
chance of an overlapping pair small. The band is not a representation
problem, so BORDERLINE is the correct verdict for those 401 pairs rather
than a placeholder.

ONE CAVEAT ON THE BASELINE, stated because it cuts against the framing above:
the `global` row here is NOT the production axis -- it is plain CLS chamfer
over 8 frames, with no corpus whitening, no static body mask and none of the
cos/bhat terms. It scores 0.5051 on the band where production env_raw scores
0.8182. So most of the env axis's power on hard pairs comes from the
WHITENING, not from raw DINOv2 similarity, which is consistent with the 23x
whitening effect measured in docs/08. The comparison against `patch` is
therefore unfair to production and generous to patch matching -- and patch
matching still loses.

import json
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMES = os.path.join(HERE, "frames")
DEV = "cuda"
# 518x294 keeps the 16:9 aspect and is a whole number of 14px patches:
# 37 x 21 = 777 tokens, so patch centres map back to real image positions.
PW, PH = 518, 294
GW, GH = PW // 14, PH // 14
RATIO = 0.90          # patch descriptors are smoother than SIFT; 0.8 kills almost everything
SIFT_RATIO = 0.75
TOPK = 3              # clips may overlap only briefly: score on the best few frame pairs

pairs = json.load(open(os.path.join(HERE, "pairs.json")))
vids = sorted({v for p in pairs for v in (p["a"], p["b"])})
print(f"{len(pairs)} pairs, {len(vids)} clips", flush=True)

# ------------------------------------------------------------------ encoders
from transformers import AutoImageProcessor, AutoModel                # noqa: E402

MODEL = "facebook/dinov2-base"
proc = AutoImageProcessor.from_pretrained(MODEL)
net = AutoModel.from_pretrained(MODEL).to(DEV).eval().half()
MEAN = torch.tensor(proc.image_mean, device=DEV).view(1, 3, 1, 1)
STD = torch.tensor(proc.image_std, device=DEV).view(1, 3, 1, 1)

# patch-centre coordinates in the resized frame, for the geometric check
yy, xx = np.meshgrid(np.arange(GH), np.arange(GW), indexing="ij")
CENTRES = np.stack([xx.ravel() * 14 + 7, yy.ravel() * 14 + 7], 1).astype(np.float32)
BODY_ROW = int(GH * 0.78)                      # below this: the wearer, mostly
KEEP_TOP = (yy.ravel() < BODY_ROW)

sift = cv2.SIFT_create(nfeatures=800)
bf = cv2.BFMatcher()


@torch.no_grad()
def encode(uuid):
    arr = np.load(os.path.join(FRAMES, uuid + ".npy"))
    t = torch.from_numpy(arr).to(DEV).permute(0, 3, 1, 2).float().div_(255)
    t = torch.nn.functional.interpolate(t, size=(PH, PW), mode="bilinear",
                                        align_corners=False)
    t = ((t - MEAN) / STD).half()
    out = net(pixel_values=t).last_hidden_state          # (F, 1+GH*GW, D)
    cls = torch.nn.functional.normalize(out[:, 0].float(), dim=-1)
    pat = torch.nn.functional.normalize(out[:, 1:].float(), dim=-1)
    return cls.cpu(), pat.cpu()


def sift_feats(uuid):
    arr = np.load(os.path.join(FRAMES, uuid + ".npy"))
    out = []
    for fr in arr:
        g = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
        kp, de = sift.detectAndCompute(g, None)
        pts = np.array([k.pt for k in kp], np.float32) if kp else np.zeros((0, 2), np.float32)
        out.append((pts, de if de is not None else np.zeros((0, 128), np.float32)))
    return out


print("encoding…", flush=True)
CLS, PAT, SIFTF = {}, {}, {}
t0 = time.time()
for i, v in enumerate(vids, 1):
    CLS[v], PAT[v] = encode(v)
    SIFTF[v] = sift_feats(v)
    if i % 20 == 0 or i == len(vids):
        print(f"  {i}/{len(vids)}  {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------- scorers
def global_chamfer(a, b):
    """Current axis: set-to-set best-match mean over global vectors."""
    S = CLS[a] @ CLS[b].T
    return float(0.5 * (S.max(1).values.mean() + S.max(0).values.mean()))


def _ransac_inliers(pa, pb, thresh=6.0):
    """Do the matches agree on ONE geometry? Affine is the permissive choice:
    a full fundamental matrix is stricter but needs 8 clean matches, and the
    band's true positives are exactly the pairs with few matches."""
    if len(pa) < 4:
        return 0
    M, mask = cv2.estimateAffinePartial2D(
        pa.reshape(-1, 1, 2), pb.reshape(-1, 1, 2), method=cv2.RANSAC,
        ransacReprojThreshold=thresh, maxIters=2000, confidence=0.995)
    return 0 if mask is None else int(mask.sum())


def patch_score(a, b, mask_body):
    """Mutual nearest neighbours between patch grids, ratio-tested, then
    geometrically verified. Reported as the mean of the best TOPK frame pairs."""
    A, B = PAT[a].to(DEV), PAT[b].to(DEV)
    keep = torch.from_numpy(KEEP_TOP).to(DEV) if mask_body else None
    scores = []
    for i in range(A.shape[0]):
        ai = A[i][keep] if mask_body else A[i]
        ci = CENTRES[KEEP_TOP] if mask_body else CENTRES
        for j in range(B.shape[0]):
            bj = B[j][keep] if mask_body else B[j]
            S = ai @ bj.T
            v2, i2 = S.topk(2, dim=1)
            fwd = i2[:, 0]
            ok = (v2[:, 1] / v2[:, 0].clamp_min(1e-6)) < RATIO      # ratio test
            back = S.argmax(0)[fwd] == torch.arange(S.shape[0], device=DEV)
            sel = (ok & back).nonzero().squeeze(1)
            if sel.numel() < 4:
                scores.append(0.0)
                continue
            pa = ci[sel.cpu().numpy()]
            pb = ci[fwd[sel].cpu().numpy()]
            scores.append(_ransac_inliers(pa, pb) / max(len(ci), 1))
    s = np.sort(np.array(scores))[::-1]
    return float(s[:TOPK].mean())


def sift_score(a, b, mask_body):
    best = []
    for pa_, da in SIFTF[a]:
        for pb_, db in SIFTF[b]:
            if len(da) < 8 or len(db) < 8:
                best.append(0.0)
                continue
            if mask_body:
                ka = pa_[:, 1] < 394 * 0.78
                kb = pb_[:, 1] < 394 * 0.78
                pa2, da2, pb2, db2 = pa_[ka], da[ka], pb_[kb], db[kb]
            else:
                pa2, da2, pb2, db2 = pa_, da, pb_, db
            if len(da2) < 8 or len(db2) < 8:
                best.append(0.0)
                continue
            m = bf.knnMatch(da2, db2, k=2)
            good = [x[0] for x in m if len(x) == 2
                    and x[0].distance < SIFT_RATIO * x[1].distance]
            if len(good) < 4:
                best.append(0.0)
                continue
            src = np.array([pa2[g.queryIdx] for g in good], np.float32)
            dst = np.array([pb2[g.trainIdx] for g in good], np.float32)
            best.append(_ransac_inliers(src, dst, thresh=8.0) / 100.0)
    s = np.sort(np.array(best))[::-1]
    return float(s[:TOPK].mean())


print("\nscoring…", flush=True)
t0 = time.time()
for k, p in enumerate(pairs, 1):
    a, b = p["a"], p["b"]
    p["global"] = global_chamfer(a, b)
    p["patch"] = patch_score(a, b, False)
    p["patch_nobody"] = patch_score(a, b, True)
    p["sift"] = sift_score(a, b, False)
    p["sift_nobody"] = sift_score(a, b, True)
    if k % 10 == 0 or k == len(pairs):
        print(f"  {k}/{len(pairs)}  {time.time() - t0:.0f}s", flush=True)

json.dump(pairs, open(os.path.join(HERE, "scored.json"), "w"), indent=1)


# ---------------------------------------------------------------- evaluation
def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum()
                  + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def grp(st, lab):
    return [p for p in pairs if p["set"] == st and p["label"] == lab]


METHODS = ["global", "patch", "patch_nobody", "sift", "sift_nobody"]
band_p, band_n = grp("band", "same"), grp("band", "different")
ss = grp("same_session", "same")
easy_p, easy_n = grp("easy", "same"), grp("easy", "different")

print("\n" + "=" * 74)
print(f"{'method':<14}{'band 3v33':>12}{'session24 v band33':>22}{'easy 6v6':>12}")
print("-" * 74)
for m in METHODS:
    a1 = auc([p[m] for p in band_p], [p[m] for p in band_n])
    a2 = auc([p[m] for p in ss], [p[m] for p in band_n])
    a3 = auc([p[m] for p in easy_p], [p[m] for p in easy_n])
    print(f"{m:<14}{a1:>12.4f}{a2:>22.4f}{a3:>12.4f}")
print("=" * 74)
print("env_raw on band 3v33 = 0.8182 (the number to beat), from the 88 labels\n")

print("group means")
for m in METHODS:
    row = " ".join(
        f"{lab}={np.mean([p[m] for p in g]):.4f}" for lab, g in
        (("band_same", band_p), ("band_diff", band_n), ("session", ss),
         ("easy_same", easy_p), ("easy_diff", easy_n)) if g)
    print(f"  {m:<14} {row}")
