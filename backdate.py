"""Backdate commits for a specific session number across the March-April plan."""
import argparse
import json
import subprocess
import sys
import os
from pathlib import Path

def run(cmd, env=None, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if check and r.returncode != 0:
        print(f"FAILED: {cmd}\n{r.stderr}", file=sys.stderr)
        sys.exit(1)
    return r.stdout.strip()

def get_changed_files():
    out = run("git status --porcelain")
    files = []
    for line in out.splitlines():
        if line.strip():
            files.append((line[:2].strip(), line[3:].strip()))
    return files

def make_commit_message(bucket, session):
    if not bucket:
        return f"chore: session {session} progress"
    paths = [p for _, p in bucket]
    if any("test" in p.lower() for p in paths):
        prefix = "test"
    elif any(p.endswith(".md") for p in paths):
        prefix = "docs"
    elif any(x in p for p in paths for x in ("requirements", "Dockerfile", "compose", ".env", "config")):
        prefix = "chore"
    else:
        prefix = "feat"
    sample = paths[0]
    return f"{prefix}: update {sample}" + (f" and {len(paths)-1} more" if len(paths) > 1 else "")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=int, required=True, help="Session number 1-30")
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()

    if not (1 <= args.session <= 30):
        sys.exit("Session must be 1-30")

    data = json.loads(Path("schedule.json").read_text())
    date_str = data["sessions"][str(args.session)]
    day_info = data["days"][date_str]
    
    files = get_changed_files()
    if not files:
        print(f"No changes to commit for session {args.session} ({date_str}).")
        return

    n_commits = max(1, day_info["n_commits"] // 2) if args.partial else day_info["n_commits"]
    timestamps = day_info["timestamps"][:n_commits]

    print(f"Session {args.session} → {date_str} ({day_info['weekday']}): {len(files)} files → {n_commits} commits")

    # Split files into n_commits buckets
    buckets = [[] for _ in range(n_commits)]
    for i, f in enumerate(files):
        buckets[i % n_commits].append(f)

    for bucket, ts in zip(buckets, timestamps):
        if not bucket:
            continue
        run("git reset")
        for status, path in bucket:
            if status == "D":
                run(f'git rm "{path}"', check=False)
            else:
                run(f'git add "{path}"')
        msg = make_commit_message(bucket, args.session)
        env = os.environ.copy()
        env["GIT_AUTHOR_DATE"] = ts
        env["GIT_COMMITTER_DATE"] = ts
        run(f'git commit -m "{msg}"', env=env)
        print(f"  ✓ {ts}  {msg}")

    remaining = get_changed_files()
    if remaining:
        last_ts = timestamps[-1]
        env = os.environ.copy()
        env["GIT_AUTHOR_DATE"] = last_ts
        env["GIT_COMMITTER_DATE"] = last_ts
        run("git add -A")
        run(f'git commit -m "chore: session {args.session} remaining"', env=env)

    next_day = date_str  # for the verify command, show same day
    print(f"\nVerify: git log --since='{date_str} 00:00' --until='{date_str} 23:59' --oneline")

if __name__ == "__main__":
    main()