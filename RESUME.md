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
| `env_sigs.tar.gz` | **the 434 DINOv2 video embeddings, 25 MB.** The expensive one: regenerating it is a full embed pass over 434 videos through presigned range reads. Untar to `env_sigs/` and `match_env.py compare` runs immediately. |

`eyeball/human_labels_all_with_pairs.csv` holds all 138 judgements with both
uuids, so both thresholds can be re-derived from raw labels.

```bash
B=s3://stage-humyn-egocentric-stereo-data/labelling_results
aws s3 cp $B/novelty_result_v2/task1_video_pairs/state/ ./state/ --recursive
cp state/manifest_435.json /tmp/manifest_435.json
tar -xzf state/env_sigs.tar.gz                  # -> env_sigs/, skips the embed pass
tar -xzf state/vlm_task_descriptions.tar.gz -C /tmp/
aws s3 cp $B/novelty_result_v2/task2_within_video/state/task2_state.tar.gz . && tar -xzf task2_state.tar.gz
```

### Media (`novelty_data_v2/handoff_2026-09-22/`)

The frames and clips behind the labelling pages. They existed only inside the
artifacts, which are fine but deletable, and rebuilding them means presigned
reads over ~200 videos.

| File | What |
|---|---|
| `media_review.tar.gz` | 59 disagreement tiles + 118 whole-clip previews |
| `media_round3_clips.tar.gz` | round-3 previews (tiles are in `eyeball/round3_tiles/`) |
| `media_doc_examples.tar.gz` | the worked-example tiles in the task1 write-up |
| `task2_chart_data.tar.gz` | the 30 fps vs 3 fps cadence series behind the task2 charts |

### Deliberately NOT backed up

| | Why |
|---|---|
| `*.mp4` at the repo root (272 MB) | a copy of a prod chunk; prod has it |
| the 8 full-rate NPZ (82 MB) | re-fetch from `prod .../model_output_fullrate/` |
| `.novelty-*` indices (20 MB) | `novelty index` rebuilds them in minutes |
| `scratchpad/place/frames/` (796 MB) | inputs to the local-matching negative result; the script regenerates them |
| `.venv/` (7.2 GB) | `uv pip install -e '.[gpu]'` |

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
`~/.claude/projects/<escaped-cwd>/<session-id>.jsonl`. **The repo has to sit at
the same absolute path on the new machine** or Claude Code will not list the
session — it finds sessions by the current working directory, not by searching.

The directory name replaces **both `/` and `_`** with `-`. Getting that wrong
is the easy mistake here:

```
/home/ec2-user/projects_abir/video-novelty
  slashes only   -> -home-ec2-user-projects_abir-video-novelty   WRONG
  slashes and _  -> -home-ec2-user-projects-abir-video-novelty   correct
```

This session:

```
path:        /home/ec2-user/projects_abir/video-novelty
dir:         ~/.claude/projects/-home-ec2-user-projects-abir-video-novelty/
session id:  65787d15-db72-45ee-9584-48d002d3ef19
size:        ~15 MB, 5551 lines, verified valid JSONL
claude ver:  2.1.278
```

On the old machine:

```bash
tar -czf ~/claude-session.tar.gz -C ~/.claude/projects ./-home-ec2-user-projects-abir-video-novelty
```

The `./` is required, not cosmetic: the directory name begins with a dash, so
without it `tar` reads it as options and fails with `invalid option -- 'e'`.

Result: ~5.4 MB, and it carries the memory files and `MEMORY.md` as well as the
transcript.

**Move it directly (scp/rsync), not through S3.** The transcript contains a
presigned S3 URL that was pasted into the conversation — a live credential
until it expires — plus access-key IDs from other pasted links. That is fine in
a home directory and on a laptop; it is not something to leave sitting in an
object store. Delete the tarball once the new machine has it.

On the new machine, with the repo cloned to the *same* path:

```bash
mkdir -p ~/.claude/projects
tar -xzf claude-session.tar.gz -C ~/.claude/projects
cd /home/ec2-user/projects_abir/video-novelty
claude --resume 65787d15-db72-45ee-9584-48d002d3ef19
```

`claude --resume` with no id lists the sessions for the current directory;
`claude --continue` picks up the most recent one.

### If the repo lives at a different path (this happened)

The resume machine cloned to `/home/ec2-user/projects/video-novelty`, not
`projects_abir/`. The session still resumed and kept writing to the OLD
directory, while the new path mapped to a fresh empty one — so memory and past
transcripts were invisible from the new working directory, and any new memory
would have been written somewhere the old name could not see.

**Do not copy the directory** — the live session keeps writing to the old one,
so the copy starts drifting immediately. Point the new name at the old store:

```bash
cd ~/.claude/projects
rmdir ./-home-ec2-user-projects-video-novelty/memory \
      ./-home-ec2-user-projects-video-novelty          # rmdir: refuses if not empty
ln -s ./-home-ec2-user-projects-abir-video-novelty \
      ./-home-ec2-user-projects-video-novelty
```

`rmdir` rather than `rm -rf` on purpose: it fails rather than deleting if that
directory turned out to hold anything. Verify writes pass through, not just
reads — write a probe file via the new name and check it appears under the old.

The alternative is to clone at the original path instead, which needs no
symlink. Either is fine; having two real directories is not.

### What resuming does and does not restore

Verified on this machine: `--resume <id>` and `-r` exist in 2.1.278, the
session file is valid, and its `sessionId` matches its filename.

It restores the **conversation** — everything said and every tool result, so
the reasoning and the numbers come back. It does **not** restore anything
outside it:

- the scratchpad (gone with the instance) — restore `state/` from S3
- the AWS sessions — `aws sso login --profile prod` again
- the MCP connectors — they reattach per machine
- the artifacts are account-level and unaffected; the four links keep working

So expect to re-run the setup in section 1 before asking it to continue work.
`--fork-session` starts a new session id from the same history if you would
rather not append to this one.

## 3. Open threads

| Thread | State |
|---|---|
| 59 venue-label disagreements | **36 of 59 reviewed**, exported to `eyeball/review_labels_partial.csv`. The remaining 23 are still only in the artifact's `db` (`labels_review`) — re-export after finishing. Partial result: of 30 reviewed where the venue label said SAME and the original answer was *different*, 15 stayed different, 4 flipped to same, 11 unsure; all 6 of the reverse cases stood. |
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
`labels_r3`, `labels_review`. Rounds 1–3 are exported to
`eyeball/human_labels_all_with_pairs.csv`, and `labels_review` is exported as
far as it goes to `eyeball/review_labels_partial.csv`. **Re-export
`labels_review` once the last 23 pairs are done** — the artifact is the only
copy of those.
