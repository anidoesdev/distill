"""Backdate commits for a specific session number across the March-April plan."""
import argparse
import json
import subprocess
import sys
import os
from pathlib import Path

def run(cmd, env=None, check=True):
    """Run a shell command. Use only for commands without paths."""
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if check and r.returncode != 0:
        print(f"FAILED: {cmd}\n{r.stderr}", file=sys.stderr)
        sys.exit(1)
    return r.stdout.strip()

def git_run(args, env=None, check=True):
    """Run a git command with args as a list. Safe for paths with dots, spaces, etc."""
    r = subprocess.run(["git"] + args, capture_output=True, text=True, env=env)
    if check and r.returncode != 0:
        print(f"FAILED: git {' '.join(args)}\n{r.stderr}", file=sys.stderr)
        sys.exit(1)
    return r.stdout.strip()

def get_changed_files():
    """Return list of (status, path) using null-terminated output for safety."""
    r = subprocess.run(
        ["git", "status", "--porcelain", "-z"],
        capture_output=True, text=True, check=True
    )
    files = []
    for entry in r.stdout.split("\0"):
        if not entry:
            continue
        status = entry[:2].strip()
        path = entry[3:]
        if path:
            files.append((status, path))
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

    # If we have fewer files than commits, cap commits at file count
    n_commits = min(n_commits, len(files))
    timestamps = timestamps[:n_commits]

    print(f"Session {args.session} → {date_str} ({day_info['weekday']}): {len(files)} files → {n_commits} commits")

    # Split files into buckets
    buckets = [[] for _ in range(n_commits)]
    for i, f in enumerate(files):
        buckets[i % n_commits].append(f)

    for bucket, ts in zip(buckets, timestamps):
        if not bucket:
            continue
        # Unstage everything first
        git_run(["reset"])
        # Stage just this bucket's files
        for status, path in bucket:
            if status == "D":
                git_run(["rm", "--", path], check=False)
            else:
                git_run(["add", "--", path])
        msg = make_commit_message(bucket, args.session)
        env = os.environ.copy()
        env["GIT_AUTHOR_DATE"] = ts
        env["GIT_COMMITTER_DATE"] = ts
        git_run(["commit", "-m", msg], env=env)
        print(f"  ✓ {ts}  {msg}")

    # Catch any leftover files
    remaining = get_changed_files()
    if remaining:
        last_ts = timestamps[-1]
        env = os.environ.copy()
        env["GIT_AUTHOR_DATE"] = last_ts
        env["GIT_COMMITTER_DATE"] = last_ts
        git_run(["add", "-A"])
        git_run(["commit", "-m", f"chore: session {args.session} remaining"], env=env)
        print(f"  ✓ {last_ts}  remaining files")

    print(f"\nVerify: git log --since='{date_str} 00:00' --until='{date_str} 23:59' --oneline")

if __name__ == "__main__":
    main()