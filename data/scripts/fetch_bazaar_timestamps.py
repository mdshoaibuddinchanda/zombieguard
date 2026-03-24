"""
fetch_bazaar_timestamps.py
Fetch first_seen timestamps from MalwareBazaar for all malicious real-world samples.
Maps short filenames (first 20 chars of SHA256) to full SHA256 → first_seen date.
Output: data/bazaar_timestamps.csv (sha256_short, sha256_full, first_seen, tags)
"""

import csv
import json
import os
import time
from pathlib import Path

import requests

PROGRESS_FILE = "data/download_progress.json"
OUTPUT_FILE = "data/bazaar_timestamps.csv"
BAZAAR_API = "https://mb-api.abuse.ch/api/v1/"
RESUME_FILE = "data/timestamp_progress.json"

# API key — set here or via environment variable MALWAREBAZAAR_API_KEY
_HARDCODED_KEY = "1ff9bb524b68df484dcdb423b08ba166146042bc86a2c5ea"
if not os.environ.get("MALWAREBAZAAR_API_KEY"):
    os.environ["MALWAREBAZAAR_API_KEY"] = _HARDCODED_KEY


def load_sha256_list() -> list[str]:
    """Load full SHA256 hashes from download progress file."""
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_resume() -> dict[str, dict]:
    """Load already-fetched timestamps to allow resuming."""
    if os.path.exists(RESUME_FILE):
        with open(RESUME_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_resume(fetched: dict[str, dict]) -> None:
    with open(RESUME_FILE, "w", encoding="utf-8") as f:
        json.dump(fetched, f)


def query_hash_info(sha256: str) -> dict | None:
    """Query MalwareBazaar for a single hash. Returns data dict or None."""
    payload = {"query": "get_info", "hash": sha256}
    headers = {"Auth-Key": "1ff9bb524b68df484dcdb423b08ba166146042bc86a2c5ea"}
    try:
        resp = requests.post(BAZAAR_API, data=payload, headers=headers, timeout=30)
        result = resp.json()
        if result.get("query_status") == "ok":
            data = result.get("data", [])
            if data:
                return data[0]
    except Exception as e:
        print(f"  Error querying {sha256[:16]}: {e}", flush=True)
    return None


def main():
    sha256_list = load_sha256_list()
    print(f"Total SHA256 hashes in progress file: {len(sha256_list)}", flush=True)

    # Filter to only those that have a corresponding file in real_world_validation
    validation_dir = Path("data/real_world_validation")
    existing_files = {f.stem for f in validation_dir.glob("*.zip")}
    print(f"Files in real_world_validation: {len(existing_files)}", flush=True)

    # Build mapping: sha256_short (first 20 chars) -> sha256_full
    sha_map = {}
    for sha in sha256_list:
        short = sha[:20]
        if short in existing_files:
            sha_map[short] = sha

    # Also handle 'recent_' prefixed files
    for fname in existing_files:
        if fname.startswith("recent_"):
            partial = fname[7:]
            for sha in sha256_list:
                if sha.startswith(partial):
                    sha_map[fname] = sha
                    break

    print(f"Matched {len(sha_map)} files to full SHA256 hashes", flush=True)

    fetched = load_resume()
    print(f"Already fetched: {len(fetched)} timestamps (resuming)", flush=True)

    # Only skip entries that were successfully fetched (have a first_seen value)
    to_fetch = [(short, full) for short, full in sha_map.items()
                if full not in fetched or not fetched[full].get("first_seen")]
    print(f"Need to fetch: {len(to_fetch)} timestamps\n", flush=True)

    for i, (short, full_sha) in enumerate(to_fetch):
        info = query_hash_info(full_sha)
        if info:
            fetched[full_sha] = {
                "sha256_short": short,
                "sha256_full": full_sha,
                "first_seen": info.get("first_seen", ""),
                "tags": ",".join(info.get("tags", []) or []),
                "file_name": info.get("file_name", ""),
                "file_type": info.get("file_type", ""),
            }
            print(f"  [{i+1}/{len(to_fetch)}] {full_sha[:16]} -> {fetched[full_sha]['first_seen']}", flush=True)
        else:
            fetched[full_sha] = {
                "sha256_short": short,
                "sha256_full": full_sha,
                "first_seen": "",
                "tags": "",
                "file_name": "",
                "file_type": "",
            }
            print(f"  [{i+1}/{len(to_fetch)}] {full_sha[:16]} -> NOT FOUND", flush=True)

        if (i + 1) % 50 == 0:
            save_resume(fetched)
            print(f"  Progress saved ({len(fetched)} total)", flush=True)

        time.sleep(0.3)

    save_resume(fetched)

    rows = list(fetched.values())
    rows.sort(key=lambda r: r.get("first_seen", "") or "")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sha256_short", "sha256_full", "first_seen", "tags", "file_name", "file_type"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {OUTPUT_FILE}", flush=True)

    with_ts = sum(1 for r in rows if r.get("first_seen"))
    without_ts = len(rows) - with_ts
    print(f"With timestamps: {with_ts}", flush=True)
    print(f"Without timestamps: {without_ts}", flush=True)

    if with_ts > 0:
        dates = sorted(r["first_seen"] for r in rows if r.get("first_seen"))
        print(f"Date range: {dates[0]} -> {dates[-1]}", flush=True)


if __name__ == "__main__":
    main()
