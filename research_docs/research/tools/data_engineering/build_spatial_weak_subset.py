import json
import pandas as pd
from pathlib import Path

SRC = Path("/workspace/fastwam_libero90/FastWAM/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot")
DST = Path("/workspace/fastwam_libero90/FastWAM/data/libero_mujoco3.3.2/libero_spatial_weak_subset_lerobot")

TARGET_TASKS = [
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    "pick up the black bowl on the ramekin and place it on the plate",
]

def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

def write_jsonl(items, p):
    with open(p, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

def main():
    episodes = load_jsonl(SRC / "meta" / "episodes.jsonl")
    ep_stats = {d["episode_index"]: d for d in load_jsonl(SRC / "meta" / "episodes_stats.jsonl")}
    info = json.loads((SRC / "meta" / "info.json").read_text())

    target_set = set(TARGET_TASKS)
    selected = [e for e in episodes if e["tasks"][0] in target_set]
    selected.sort(key=lambda e: e["episode_index"])
    print(f"selected {len(selected)} episodes out of {len(episodes)}")
    by_task = {}
    for e in selected:
        by_task[e["tasks"][0]] = by_task.get(e["tasks"][0], 0) + 1
    for t, c in by_task.items():
        print(f"  {c:4d}  {t}")

    chunks_size = info["chunks_size"]
    n = len(selected)
    assert n <= chunks_size

    unique_tasks = sorted(set(e["tasks"][0] for e in selected))
    task_to_index = {t: i for i, t in enumerate(unique_tasks)}

    (DST / "meta").mkdir(parents=True, exist_ok=True)
    (DST / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    for vk in video_keys:
        (DST / "videos" / "chunk-000" / vk).mkdir(parents=True, exist_ok=True)

    new_episodes = []
    new_ep_stats = []
    total_frames = 0
    global_row_index = 0
    for new_idx, e in enumerate(selected):
        old_idx = e["episode_index"]
        old_chunk = old_idx // chunks_size
        new_task_idx = task_to_index[e["tasks"][0]]

        old_parquet = SRC / info["data_path"].format(episode_chunk=old_chunk, episode_index=old_idx)
        new_parquet = DST / info["data_path"].format(episode_chunk=0, episode_index=new_idx)
        df = pd.read_parquet(old_parquet)
        n_rows = len(df)
        df["episode_index"] = new_idx
        df["task_index"] = new_task_idx
        df["index"] = range(global_row_index, global_row_index + n_rows)
        df.to_parquet(new_parquet)
        global_row_index += n_rows

        for vk in video_keys:
            old_video = SRC / info["video_path"].format(episode_chunk=old_chunk, video_key=vk, episode_index=old_idx)
            new_video = DST / info["video_path"].format(episode_chunk=0, video_key=vk, episode_index=new_idx)
            new_video.symlink_to(old_video.resolve())

        new_episodes.append({
            "episode_index": new_idx,
            "tasks": e["tasks"],
            "length": e["length"],
        })
        total_frames += e["length"]

        stat_entry = ep_stats[old_idx]
        new_ep_stats.append({
            "episode_index": new_idx,
            "stats": stat_entry["stats"],
        })

    write_jsonl(new_episodes, DST / "meta" / "episodes.jsonl")
    write_jsonl(new_ep_stats, DST / "meta" / "episodes_stats.jsonl")
    write_jsonl(
        [{"task_index": i, "task": t} for t, i in task_to_index.items()],
        DST / "meta" / "tasks.jsonl",
    )

    new_info = dict(info)
    new_info["total_episodes"] = n
    new_info["total_frames"] = total_frames
    new_info["total_videos"] = n * len(video_keys)
    new_info["total_tasks"] = len(unique_tasks)
    new_info["total_chunks"] = 1
    new_info["splits"] = {"train": f"0:{n}"}
    (DST / "meta" / "info.json").write_text(json.dumps(new_info, indent=4))

    print(f"\nwrote filtered dataset to {DST}")
    print(f"total_episodes={n} total_frames={total_frames} total_tasks={len(unique_tasks)}")

if __name__ == "__main__":
    main()
