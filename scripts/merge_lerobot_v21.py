"""Merge two LeRobot v2.1 datasets into one, without going through v3.0.

openpi pins a v2.1-era lerobot, and lerobot's v2.1 -> v3.0 conversion is one-way
(there is no downgrade script), so merging has to happen at v2.1.

The first source keeps its numbering, which makes its data files byte-identical
in the output; they are hardlinked rather than rewritten. Only the second
source's files are rewritten, with episode_index / index / task_index offset.

Assumes both datasets are image-based (no videos/ tree). Verified up front.

Usage:
    python merge_lerobot_v21.py --out /path/to/merged
    python merge_lerobot_v21.py --out /path/to/merged --copy   # copy, don't hardlink
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

DATA_PATH = "data/chunk-{chunk:03d}/episode_{ep:06d}.parquet"
CHUNKS_SIZE = 1000
# Columns that encode a position in the dataset and must be renumbered on offset.
REINDEXED = ("episode_index", "index", "task_index")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def validate(infos, roots):
    """Fail loudly on anything that would silently corrupt the merge."""
    ref, ref_root = infos[0], roots[0]
    for info, root in zip(infos[1:], roots[1:], strict=True):
        for key in ("codebase_version", "fps", "robot_type"):
            if info.get(key) != ref.get(key):
                raise ValueError(f"{key} mismatch: {ref_root.name}={ref.get(key)} {root.name}={info.get(key)}")
        if info["features"] != ref["features"]:
            raise ValueError(f"feature schema mismatch between {ref_root.name} and {root.name}")
        if info.get("total_videos", 0) != 0 or (root / "videos").is_dir():
            raise ValueError(f"{root.name} has videos; this script only handles image-based datasets")
    if ref.get("codebase_version") != "v2.1":
        raise ValueError(f"expected v2.1 datasets, got {ref.get('codebase_version')}")


def stats_of(values):
    """Per-feature stats in the same shape lerobot writes for scalar columns."""
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": [float(arr.min())],
        "max": [float(arr.max())],
        "mean": [float(arr.mean())],
        "std": [float(arr.std())],
        "count": [int(arr.size)],
    }


def merge(roots, out, link=True):
    infos = [json.load(open(r / "meta/info.json")) for r in roots]
    validate(infos, roots)

    # --- tasks: dedupe by task string; first dataset's indices are preserved ---
    task_to_new = {}
    merged_tasks = []
    per_ds_task_map = []
    for root in roots:
        rows = sorted(read_jsonl(root / "meta/tasks.jsonl"), key=lambda r: r["task_index"])
        mapping = {}
        for row in rows:
            task = row["task"]
            if task not in task_to_new:
                task_to_new[task] = len(merged_tasks)
                merged_tasks.append({"task_index": task_to_new[task], "task": task})
            mapping[row["task_index"]] = task_to_new[task]
        per_ds_task_map.append(mapping)

    out.mkdir(parents=True, exist_ok=True)

    merged_episodes = []
    merged_stats = []
    ep_offset = 0
    idx_offset = 0

    for ds_i, root in enumerate(roots):
        episodes = sorted(read_jsonl(root / "meta/episodes.jsonl"), key=lambda r: r["episode_index"])
        stats = {s["episode_index"]: s["stats"] for s in read_jsonl(root / "meta/episodes_stats.jsonl")}
        tmap = per_ds_task_map[ds_i]
        # `index` is a global running counter within its own dataset, so it
        # shifts by a constant (the frames already emitted), not per episode.
        ds_idx_offset = idx_offset
        identity = ep_offset == 0 and ds_idx_offset == 0 and all(k == v for k, v in tmap.items())

        desc = f"[{ds_i + 1}/{len(roots)}] {root.name}" + (" (link)" if identity and link else "")
        for ep in tqdm(episodes, desc=desc):
            old_ep = ep["episode_index"]
            new_ep = old_ep + ep_offset
            src = root / DATA_PATH.format(chunk=old_ep // CHUNKS_SIZE, ep=old_ep)
            dst = out / DATA_PATH.format(chunk=new_ep // CHUNKS_SIZE, ep=new_ep)
            dst.parent.mkdir(parents=True, exist_ok=True)

            ep_stats = json.loads(json.dumps(stats[old_ep]))  # deep copy

            if identity:
                # Numbering already correct -> file content is unchanged.
                if dst.exists():
                    dst.unlink()
                if link:
                    dst.hardlink_to(src)
                else:
                    shutil.copy2(src, dst)
            else:
                table = pq.read_table(src)
                new_cols = {
                    "episode_index": [new_ep] * table.num_rows,
                    "index": [i + ds_idx_offset for i in table.column("index").to_pylist()],
                    "task_index": [tmap[t] for t in table.column("task_index").to_pylist()],
                }
                for name, values in new_cols.items():
                    i = table.schema.get_field_index(name)
                    field = table.schema.field(i)
                    table = table.set_column(i, field, pa.array(values, type=field.type))
                pq.write_table(table, dst)
                # Stats for renumbered columns are now stale; recompute from the
                # actual new values. All other features are untouched by merging.
                for name in REINDEXED:
                    ep_stats[name] = stats_of(new_cols[name])

            merged_episodes.append(
                {"episode_index": new_ep, "tasks": ep["tasks"], "length": ep["length"]}
            )
            merged_stats.append({"episode_index": new_ep, "stats": ep_stats})
            idx_offset += ep["length"]

        ep_offset += len(episodes)

    total_episodes = len(merged_episodes)
    total_frames = idx_offset

    info = json.loads(json.dumps(infos[0]))
    info.update(
        total_episodes=total_episodes,
        total_frames=total_frames,
        total_tasks=len(merged_tasks),
        total_videos=0,
        total_chunks=math.ceil(total_episodes / CHUNKS_SIZE),
        splits={"train": f"0:{total_episodes}"},
    )

    write_jsonl(out / "meta/episodes.jsonl", merged_episodes)
    write_jsonl(out / "meta/episodes_stats.jsonl", merged_stats)
    write_jsonl(out / "meta/tasks.jsonl", merged_tasks)
    (out / "meta/info.json").write_text(json.dumps(info, indent=4))

    print(f"\nMerged -> {out}")
    print(f"  episodes: {total_episodes}  frames: {total_frames}  tasks: {len(merged_tasks)}")


if __name__ == "__main__":
    base = Path("/mnt/data/vhoangth2/datasets")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=[base / "ryanhoangt__libero_90_lerobot_v21", base / "ryanhoangt__libero-icl-finetune"],
        help="Source dataset dirs, in output order. The first keeps its numbering.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of hardlinking.")
    args = parser.parse_args()
    merge(args.roots, args.out, link=not args.copy)
