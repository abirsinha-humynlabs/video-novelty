# 05 — Coverage selection

## The objective

```
f(S) = Σ_{v ∈ corpus}  max_{s ∈ S}  sim(v, s)
```

In words: *how well is every clip in my corpus represented by something I
selected.* Maximise it subject to `|S| ≤ budget`.

This is **facility location**. It is monotone (adding a clip never hurts) and
**submodular** (adding a clip to a small selection helps at least as much as
adding it to a large one). Those two properties buy you:

- **A guarantee.** Greedy is within `(1 − 1/e) ≈ 63%` of the optimum. Threshold
  dedup has no guarantee at all — its result depends on ingest order.
- **A curve.** Marginal gains fall monotonically, so plotting them gives you the
  diminishing-returns curve for your dataset, computed from your data rather
  than guessed.
- **Dedup for free.** A clip whose marginal gain is ≈0 is redundant, and you
  know *how* redundant rather than just which side of a line it landed on.

## Why not threshold dedup

| | threshold dedup | facility location |
|---|---|---|
| depends on ingest order | yes | no |
| keeps a rare clip near a common one | no — deletes it | yes — high marginal gain |
| tells you where to stop collecting | no | yes, that's the knee |
| answers "which 200 of 2000 do I label" | no | that's literally the query |
| guarantee | none | (1 − 1/e) |

The failure that matters most in practice is the second row. A rare event that
happens to occur in a visually ordinary clip is exactly the sample your model
needs and exactly the one an ε-ball deletes.

## The implementation

`select.facility_location_greedy(S, budget, preselected=None)`.

**Lazy (accelerated) greedy.** Because `f` is submodular, a candidate's gain can
only shrink as the selection grows. So a stale gain sitting at the top of a
max-heap, which still tops the heap after re-evaluation, is provably the true
best — no need to re-score everything each round. This turns O(n²) per pick into
something much closer to O(n log n).

**`preselected`** lets you extend an existing dataset rather than re-choosing it
from scratch: pass the indices you have already labelled and greedy will pick
what best complements them.

### The similarity matrix

`Index.similarity_matrix(kind=...)` builds `S` from calibrated percentiles
scaled to [0, 1], not raw cosines. This matters more than it looks: raw cosines
squashed into [0.90, 1.00] make every clip look equally well covered, so every
marginal gain is tiny and the ordering is noise. `kind` selects which axis
drives selection:

- `env` — maximise *scene* diversity. Use when you need coverage of places.
- `task` — maximise *behaviour* diversity. Usually what a manipulation policy
  needs.
- `fused` — the mean of both. The default.

## Reading the output

```
  #     gain  cum.cov  clip
  1    9.960   62.25%  clipC_v5_400-460.mp4[15-45s]
  2    2.414   77.34%  clipD_front_040-100.mp4[15-45s]
  3    1.680   87.84%  clipA_040-100.mp4[15-45s]
  4    0.722   92.35%  clipB_100-160.mp4[30-60s]
  5    0.412   94.93%  clipD_front_040-100.mp4[30-60s]
  6    0.362   97.19%  clipC_v5_400-460.mp4[30-60s]
  7    0.346   99.35%  clipB_100-160.mp4[0-30s]
  8    0.070   99.79%  clipA_040-100.mp4[30-60s]
  9    0.034  100.00%  clipA_040-100.mp4[0-30s]
 10    0.000  100.00%  clipB_100-160.mp4[15-45s]
 ...
knee at 7 clip(s): after this, each extra clip adds < 1.0% of total coverage.
```

Two things to notice.

**The first four picks come from four different source clips.** The selector
exhausts the distinct material before it takes a second window from anything.
That is the submodular guarantee doing its job, and it is the behaviour you
cannot get from thresholding.

**Coverage saturates at 9 of 16 windows.** Picks 10–16 add exactly zero. On this
corpus, 44% of the footage is carrying no information the other 56% doesn't
already have.

## The knee

`SelectionResult.knee(tol=0.01)` returns the first pick after which each
additional clip contributes less than 1% of total coverage.

This is the number that answers *"monotonous task data stops helping after a
while — when?"*. It is not a rule of thumb; it is measured on your data, and it
moves when your data moves. Two ways to use it:

- **Collection planning.** Re-run the index weekly. If the knee stops moving
  while hours keep accumulating, the new footage is not adding anything and the
  camera should be pointed somewhere else.
- **Label budgeting.** Label to the knee first. Everything past it has a
  measured, tiny marginal contribution.

## The ingest gate

`novelty gate new.mp4 --index .novelty --threshold 0.97`

Nearest-neighbour based, deliberately simple, because at ingest time you usually
do not have the corpus in RAM. Exit code 1 on rejection so it drops into a
pipeline:

```bash
for f in incoming/*.mp4; do
  novelty gate "$f" --index .novelty && mv "$f" accepted/ || mv "$f" rejected/
done
```

**For batch curation, prefer `select`.** The gate answers "is this close to
something I have", which is the ε-ball question with all its flaws. `select`
answers "does this improve my coverage", which is the right one. Use the gate
when clips arrive one at a time and you need an answer now; use `select` when
you have a pile and time to think.

→ Next: [06 — Running on GPU](06-running-on-gpu.md)
