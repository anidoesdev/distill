"""Generate a realistic March-April 2026 commit schedule.

62 calendar days, 30 build sessions, 3 gap days, 2 sprint days.
Weekends lighter, weekdays normal, sprints occasionally heavy.
"""
import json
import random
from datetime import datetime, timedelta

random.seed(42)

# Build the calendar: March 1 - April 30, 2026
start = datetime(2026, 3, 1)
calendar = [start + timedelta(days=i) for i in range(62)]

def is_weekend(d):
    return d.weekday() >= 5

# Pick 3 gap days (no commits) — bias toward weekends for realism
weekend_indices = [i for i, d in enumerate(calendar) if is_weekend(d)]
weekday_indices = [i for i, d in enumerate(calendar) if not is_weekend(d)]
gap_days = set(random.sample(weekend_indices, 2) + random.sample(weekday_indices, 1))

# Pick 2 sprint days (heavy commit days) — bias toward weekdays
non_gap_weekdays = [i for i in weekday_indices if i not in gap_days]
sprint_days = set(random.sample(non_gap_weekdays, 2))

# 30 session days from the remaining 59 working days, evenly spaced-ish
working_days = [i for i in range(62) if i not in gap_days]
# Pick 30 working days for actual sessions; the other 29 get "light maintenance" commits
session_indices = sorted(random.sample(working_days, 30))
session_map = {idx: session_num + 1 for session_num, idx in enumerate(session_indices)}

def commit_count(i, d):
    if i in gap_days:
        return 0
    if i in sprint_days:
        return random.randint(7, 9)
    if i in session_indices:
        # Real session day: 3-6 commits weekdays, 1-3 weekends
        return random.randint(1, 3) if is_weekend(d) else random.randint(3, 6)
    # Non-session working day: light maintenance, README tweaks, docs
    return random.randint(1, 2) if random.random() < 0.6 else 0

schedule = {}
for i, day in enumerate(calendar):
    n = commit_count(i, day)
    
    hour_pool = (
        list(range(10, 13)) * 2 +
        list(range(14, 18)) * 3 +
        list(range(19, 23)) * 3 +
        [23, 0, 1]
    )
    
    timestamps = []
    if n > 0:
        hours = random.sample(hour_pool, min(n, len(hour_pool)))
        if len(hours) < n:
            hours += random.choices(hour_pool, k=n - len(hours))
        hours.sort()
        for h in hours:
            ts = day.replace(hour=h % 24, minute=random.randint(0, 59), second=random.randint(0, 59))
            timestamps.append(ts.isoformat())
    
    schedule[day.strftime("%Y-%m-%d")] = {
        "calendar_index": i,
        "weekday": day.strftime("%A"),
        "is_weekend": is_weekend(day),
        "is_gap": i in gap_days,
        "is_sprint": i in sprint_days,
        "is_session": i in session_indices,
        "session_number": session_map.get(i),
        "n_commits": n,
        "timestamps": timestamps,
    }

# Also build a session_number -> date lookup for the tutor
session_lookup = {
    info["session_number"]: date
    for date, info in schedule.items()
    if info["session_number"] is not None
}

with open("schedule.json", "w") as f:
    json.dump({"days": schedule, "sessions": session_lookup}, f, indent=2)

total = sum(d["n_commits"] for d in schedule.values())
print(f"Generated 62-day schedule. Total commits: {total}")
print(f"Gap days: {sorted([calendar[i].strftime('%Y-%m-%d') for i in gap_days])}")
print(f"Sprint days: {sorted([calendar[i].strftime('%Y-%m-%d') for i in sprint_days])}")
print(f"Session day count: {len(session_indices)}, maintenance day count: {len([i for i in working_days if i not in session_indices])}")
print(f"\nSession 1 → {session_lookup[1]}, Session 30 → {session_lookup[30]}")