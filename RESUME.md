# Resuming this work on another machine

Written 2026-09-22 while handing off. Two separate things to move: **the repo
and its state**, and **the Claude Code conversation**. They are independent —
you can have either without the other.

## 1. The work itself

```bash
git clone https://github.com/abirsinha-humynlabs/video-novelty.git
cd video-novelty
uv venv && uv pip install -e '.[gpu]'      # see WORK.md section 9 for the ffmpeg note
```

Read **WORK.md section 000 first** (task1, delivered), then section 00 (task2,
parked on frame rate).

### Nothing big lives in git, by design

The pair CSVs and all labelling artefacts are on S3, not in the repo — they
carry columns derived from the customer QA export, and every threshold change
rewrote all 13,964 rows. Canonical location:

```
s3://stage-humyn-egocentric-stereo-data/labelling_results/novelty_result_v2/
  task1_video_pairs/     env_pairs*.csv, eyeball/, state/, README.md
  task2_within_video/    one CSV per episode, README.md
```

Each prefix has its own `README.md`. Start there.

### State needed to re-run anything (`task1_video_pairs/state/`)

Reconstructed here because it was produced in a session scratchpad and would
otherwise be gone:

| File | Why it matters |
|---|---|
| `manifest_435.json` | episode uuid to S3 video URI. **Every script that touches video needs this.** Scripts expect it at `/tmp/manifest_435.json`. |
| `round1_pairs.csv` | round-1 pairs with uuids |
| `round2_pairs.json`, `round3_pairs.json` | slot to uuid for rounds 2 and 3 — without these the labels cannot be traced to episodes |
| `round2_labels.json`, `round3_labels.json` | the raw answers |
| `review_disagreements.json` | the 59 venue-label disagreement pairs |
| `vlm_key_map.json` | episode uuid to VLM S3 key |
| `vlm_task_descriptions.tar.gz` | the 433 VLM descriptions (also re-fetchable) |

`eyeball/human_labels_all_with_pairs.csv` holds all 138 judgements with both
uuids, so both thresholds can be re-derived from raw labels.

```bash
aws s3 cp s3://stage-humyn-egocentric-stereo-data/labelling_results/novelty_result_v2/task1_video_pairs/state/ ./state/ --recursive
cp state/manifest_435.json /tmp/manifest_435.json
tar -xzf state/vlm_task_descriptions.tar.gz -C /tmp/
```

### Access

Two separate credentials, and both expired at least once during the last
session:

- **prod, read-only, SSO**: `aws sso login --profile prod`. Needed for source
  video, hand-tracking NPZ and the VLM judgment. Sessions are short — expect to
  re-login. **Prod is read-only; never write to it.**
- **stage**: the EC2 instance role, no login. The only writable prefixes are
  `novelty_result_v2/` and `novelty_data_v2/`, enforced in code by
  `check_s3_destination` in `scripts/report_csv.py`.

### Rebuild the task1 output from scratch

```bash
python scripts/match_env.py embed   --manifest /tmp/manifest_435.json --out env_sigs
python scripts/match_env.py compare --sigs env_sigs --out env_pairs.csv
python scripts/task_axis.py env_pairs.csv env_pairs_by_industry.csv   # uploads to S3
```

## 2. The conversation

Claude Code stores each session as one JSONL under
`~/.claude/projects/<escaped-cwd>/<session-id>.jsonl`. The directory name is
the working directory with `/` replaced by `-`, so **the repo has to sit at the
same absolute path on the new machine** or Claude Code will not list the
session.

This session:

```
path:       /home/ec2-user/projects_abir/video-novelty
dir:        ~/.claude/projects/-home-ec2-user-projects-abir-video-novelty/
session id: 65787d15-db72-45ee-9584-48d002d3ef19
size:       ~14 MB
```

On the old machine:

```bash
tar -czf ~/claude-session.tar.gz -C ~/.claude/projects -home-ec2-user-projects-abir-video-novelty
```

On the new machine, with the repo cloned to the *same* path:

```bash
mkdir -p ~/.claude/projects
tar -xzf claude-session.tar.gz -C ~/.claude/projects
cd /home/ec2-user/projects_abir/video-novelty
claude --resume 65787d15-db72-45ee-9584-48d002d3ef19
```

`claude --resume` with no id lists the sessions for the current directory;
`claude --continue` picks up the most recent one.

If the repo must live at a different path, rename the directory under
`~/.claude/projects/` to match the new path in the same `/` to `-` form.

**The transcript is a conversation, not a backup.** It does not carry the
scratchpad, so restore `state/` from S3 regardless.

## 3. Open threads

| Thread | State |
|---|---|
| 59 venue-label disagreements | Under review at the artifact below. Results are in its `db` under collection `labels_review`, not in any file yet. |
| task2 | Parked on 3 fps hand tracking. Poll both `model_output/` and `model_output_fullrate/`. |
| VLM task axis | Measured, deliberately not adopted — see `scripts/vlm_task.py`. Use `task_name` for episodes outside the 435, where the QA column does not exist. |
| Pushing | Git here authenticates through the VS Code askpass bridge; when the IDE connection drops, pushes fail with "No anonymous write access". |

### Artifacts (private — share from each page's Share menu)

| Page | URL |
|---|---|
| Task 1 write-up | https://claude.ai/code/artifact/b8d0c605-2b04-4c46-b736-eb2dca3f6be5 |
| Task 2 write-up | https://claude.ai/code/artifact/66c0ad9b-1c45-468d-99b1-154ac8837d9c |
| Labelling rounds 1–3 | https://claude.ai/artifact/PgiK6Ge59by3vGafr7L9fM |
| Venue-label disagreements | https://claude.ai/artifact/V3k2tCeySBcxrCVYhGV1V8 |

Label data lives in each page's `db`: collections `labels`, `labels_r2`,
`labels_r3`, `labels_review`. Rounds 1–3 are already exported to
`eyeball/human_labels_all_with_pairs.csv`; **`labels_review` is not**, so
export it before deleting anything.
