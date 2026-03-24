"""
fetch_timestamps_v2.py - Simplified, robust version.
Fetches MalwareBazaar first_seen timestamps for all real-world samples.
"""
import csv
import json
import sys
import time
from pathlib import Path

import requests

# Log to file since conda run swallows stdout on Windows
_log = open("data/fetch_timestamps.log", "w", buffering=1, encoding="utf-8")

def log(msg):
    print(msg)
    _log.write(msg + "\n")
    _log.flush()

API_KEY = "1ff9bb524b68df484dcdb423b08ba166146042bc86a2c5ea"
BAZAAR_API = "https://mb-api.abuse.ch/api/v1/"
HEADERS = {"Auth-Key": API_KEY}

PROGRESS_FILE = "data/timestamp_progress.json"
OUTPUT_FILE = "data/bazaar_timestamps.csv"


def load_progress():
    try:
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_progress(data):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f)


def fetch_info(sha256):
    try:
        r = requests.post(BAZAAR_API, data={"query": "get_info", "hash": sha256},
                          headers=HEADERS, timeout=20)
        j = r.json()
        if j.get("query_status") == "ok" and j.get("data"):
            return j["data"][0]
    except Exception as e:
        log(f"  ERROR {sha256[:12]}: {e}")
    return None


def main():
    # Load all SHA256 hashes
    with open("data/download_progress.json") as f:
        all_sha = json.load(f)
    log(f"SHA256 list: {len(all_sha)}")

    # Get files in validation dir
    val_dir = Path("data/real_world_validation")
    existing = {f.stem: f for f in val_dir.glob("*.zip")}
    log(f"Validation files: {len(existing)}")

    # Build work list: (filename_stem, full_sha256)
    work = []
    sha_set = set(all_sha)
    for stem, _ in existing.items():
        # Try matching by first 20 chars
        matched = None
        for sha in all_sha:
            if sha[:20] == stem or sha[:len(stem)] == stem:
                matched = sha
                break
        if matched:
            work.append((stem, matched))

    log(f"Matched: {len(work)} files")

    # Load progress - only skip entries WITH a timestamp
    done = load_progress()
    done_with_ts = {sha for sha, v in done.items() if v.get("first_seen")}
    log(f"Already done with timestamps: {len(done_with_ts)}")

    todo = [(stem, sha) for stem, sha in work if sha not in done_with_ts]
    log(f"To fetch: {len(todo)}")

    if not todo:
        print("Nothing to fetch!")
    else:
        for i, (stem, sha) in enumerate(todo):
            info = fetch_info(sha)
            if info:
                done[sha] = {
                    "sha256_short": stem,
                    "sha256_full": sha,
                    "first_seen": info.get("first_seen", ""),
                    "tags": ",".join(info.get("tags") or []),
                    "file_name": info.get("file_name", ""),
                }
                ts = done[sha]["first_seen"]
            else:
                done[sha] = {"sha256_short": stem, "sha256_full": sha,
                             "first_seen": "", "tags": "", "file_name": ""}
                ts = "NOT FOUND"

            log(f"  [{i+1}/{len(todo)}] {sha[:16]} -> {ts}")

            if (i + 1) % 100 == 0:
                save_progress(done)
                log(f"  Saved progress: {len(done)} total")

            time.sleep(0.3)

        save_progress(done)

    # Write CSV
    rows = list(done.values())
    rows.sort(key=lambda r: r.get("first_seen") or "")
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sha256_short", "sha256_full", "first_seen", "tags", "file_name"])
        w.writeheader()
        w.writerows(rows)

    with_ts = sum(1 for r in rows if r.get("first_seen"))
    log(f"\nDone. {len(rows)} rows saved to {OUTPUT_FILE}")
    log(f"With timestamps: {with_ts} / {len(rows)}")
    if with_ts:
        dates = sorted(r["first_seen"] for r in rows if r.get("first_seen"))
        log(f"Date range: {dates[0]} -> {dates[-1]}")


if __name__ == "__main__":
    main()
