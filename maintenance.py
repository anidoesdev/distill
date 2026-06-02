"""Make a small maintenance commit on a non-session day."""
import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

MAINTENANCE_EDITS = [
    ("README.md", "\n<!-- updated -->\n"),
    (".gitignore", "\n# misc\n"),
    ("notes.md", f"\n- {datetime.now().strftime('%Y-%m-%d')}: misc notes\n"),
]

def run(cmd, env=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    data = json.loads(Path("schedule.json").read_text())
    day_info = data["days"].get(args.date)
    if not day_info:
        sys.exit(f"Date {args.date} not in schedule")
    if day_info["n_commits"] == 0:
        print(f"Gap day, skipping.")
        return

    for ts in day_info["timestamps"]:
        path, snippet = random.choice(MAINTENANCE_EDITS)
        Path(path).touch()
        with open(path, "a") as f:
            f.write(snippet)
        run(f'git add "{path}"')
        env = os.environ.copy()
        env["GIT_AUTHOR_DATE"] = ts
        env["GIT_COMMITTER_DATE"] = ts
        msg = random.choice(["docs: minor update", "chore: tidy", "docs: notes", "chore: bump"])
        run(f'git commit -m "{msg}"', env=env)
        print(f"  ✓ {ts}  {msg}")

if __name__ == "__main__":
    main()