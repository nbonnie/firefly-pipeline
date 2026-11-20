#!/usr/bin/env python3
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np

from local_contrast import ring_contrast

MAX_EXTRAPOLATION_SPEED_PX = 2.0
MIN_POINTS_FOR_VELOCITY = 4
SUPPORT_FRAC = 0.2

_BAR = 50
def _progress(i, total, tag=''):
    total = max(1, int(total or 1))
    i = min(i, total)
    frac = min(1.0, max(0.0, i / float(total)))
    bar  = '=' * int(frac * _BAR) + ' ' * (_BAR - int(frac * _BAR))
    sys.stdout.write(f'\r{tag} [{bar}] {int(frac*100):3d}%')
    sys.stdout.flush()
    if i >= total: sys.stdout.write('\n')

def _read_csv_rows(csv_path: Path) -> List[dict]:
    with csv_path.open('r', newline='') as f:
        return [dict(row) for row in csv.DictReader(f)]

def _write_csv_rows(csv_path: Path, rows: List[dict], cols: List[str]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='') as f:
        wri = csv.DictWriter(f, fieldnames=cols)
        wri.writeheader()
        for r in rows:
            wri.writerow({k: r.get(k, '') for k in cols})

def _is_center(row: dict) -> bool:
    return str(row.get('xy_semantics', '')).lower() == 'center'

def _center(row: dict) -> Tuple[float, float]:
    x = float(row['x']); y = float(row['y'])
    if _is_center(row):
        return x, y
    return x + float(row.get('w', 10)) / 2.0, y + float(row.get('h', 10)) / 2.0

def _link_streaks(dets: List[tuple], radius_px: float, max_gap: int) -> List[List[tuple]]:
    by_frame: Dict[int, List[tuple]] = defaultdict(list)
    for d in dets:
        by_frame[d[1]].append(d)

    streaks: List[List[tuple]] = []
    open_ids: List[int] = []
    for f in sorted(by_frame):
        open_ids = [s for s in open_ids if f - streaks[s][-1][1] - 1 <= max_gap]
        cur = by_frame[f]
        taken, used = set(), set()
        if open_ids:
            last = np.array([streaks[s][-1][2:4] for s in open_ids], dtype=np.float32)
            pts = np.array([d[2:4] for d in cur], dtype=np.float32)
            dist = np.hypot(pts[:, None, 0] - last[None, :, 0], pts[:, None, 1] - last[None, :, 1])
            pi, si = np.nonzero(dist <= radius_px)
            for k in np.argsort(dist[pi, si], kind='stable'):
                p, s = int(pi[k]), int(si[k])
                if p in taken or s in used:
                    continue
                taken.add(p); used.add(s)
                streaks[open_ids[s]].append(cur[p])
        for p, d in enumerate(cur):
            if p not in taken:
                streaks.append([d])
                open_ids.append(len(streaks) - 1)
    return streaks

def _sample_positions(streak: List[tuple], grow_frames: int, last_frame: int):
    fs = np.array([d[1] for d in streak], dtype=np.float64)
    xs = np.array([d[2] for d in streak], dtype=np.float64)
    ys = np.array([d[3] for d in streak], dtype=np.float64)
    start = max(0, int(fs[0]) - grow_frames)
    end = min(last_frame, int(fs[-1]) + grow_frames)
    fr = np.arange(start, end + 1, dtype=np.float64)
    px = np.interp(fr, fs, xs)
    py = np.interp(fr, fs, ys)

    if len(fs) >= MIN_POINTS_FOR_VELOCITY and fs[-1] > fs[0]:
        vx = float(np.clip(np.polyfit(fs, xs, 1)[0], -MAX_EXTRAPOLATION_SPEED_PX, MAX_EXTRAPOLATION_SPEED_PX))
        vy = float(np.clip(np.polyfit(fs, ys, 1)[0], -MAX_EXTRAPOLATION_SPEED_PX, MAX_EXTRAPOLATION_SPEED_PX))
        before = fr < fs[0]
        after = fr > fs[-1]
        px[before] = xs[0] + vx * (fr[before] - fs[0])
        py[before] = ys[0] + vy * (fr[before] - fs[0])
        px[after] = xs[-1] + vx * (fr[after] - fs[-1])
        py[after] = ys[-1] + vy * (fr[after] - fs[-1])
    return start, px, py

def _flash_runs(con: np.ndarray, grow_contrast: float, max_gap: int) -> List[Tuple[int, int]]:
    idx = np.flatnonzero(con >= grow_contrast)
    if not len(idx):
        return []
    cuts = np.flatnonzero(np.diff(idx) > max_gap + 1)
    starts = np.r_[idx[0], idx[cuts + 1]]
    ends = np.r_[idx[cuts], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))

def _curve_shape(sig: np.ndarray, a: int, b: int, margin: int) -> Tuple[int, float]:
    lo = max(0, a - margin)
    hi = min(len(sig) - 1, b + margin)
    seg = np.clip(sig[lo:hi + 1], 0, None)
    ip = (a - lo) + int(np.argmax(seg[a - lo:b - lo + 1]))
    peak = float(seg[ip])
    if peak <= 0:
        return 0, 1.0

    def _run(th):
        i = j = ip
        while i > 0 and seg[i - 1] >= th:
            i -= 1
        while j < len(seg) - 1 and seg[j + 1] >= th:
            j += 1
        return i, j

    i, j = _run(peak / 2.0)
    si, sj = _run(peak * SUPPORT_FRAC)
    outside = np.r_[seg[:si], seg[sj + 1:]]
    base = float(np.median(outside)) / peak if len(outside) >= 3 else 1.0
    return j - i + 1, base

def stage8_8_grow_and_filter_streaks(
    *,
    orig_video_path: Path,
    csv_path: Path,
    report_csv_path: Optional[Path] = None,
    seed_contrast: Optional[float] = 30.0,
    grow_contrast: float = 10.0,
    link_radius_px: float = 12.0,
    grow_frames: int = 15,
    max_frame_gap: int = 2,
    min_width_frames: int = 2,
    max_base_frac: float = 0.35,
    max_streak_seconds: float = 3.0,
    dedupe_px: float = 4.0,
    box_w: int = 10,
    box_h: int = 10,
    max_frames: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[int, int, int]:
    """
    Links firefly rows into streaks, grows each streak in time down to grow_contrast,
    and keeps only flashes that are bright enough, short enough, wide enough and not flat.

    Returns: (fireflies_before, fireflies_after, rows_added_by_growing)
    """
    rows = _read_csv_rows(csv_path)
    if not rows:
        if verbose:
            print("[stage8.8] No rows; nothing to do.")
        return (0, 0, 0)

    cols = list(rows[0].keys())
    has_class = 'class' in rows[0]

    dets: List[tuple] = []
    for idx, r in enumerate(rows):
        if has_class and r.get('class') != 'firefly':
            continue
        try:
            f = int(r['frame'])
            if max_frames is not None and f >= max_frames:
                continue
            cx, cy = _center(r)
        except Exception:
            continue
        dets.append((idx, f, cx, cy))
    if not dets:
        if verbose:
            print("[stage8.8] No firefly rows; nothing to do.")
        return (0, 0, 0)
    dets.sort(key=lambda d: d[1])

    cap = cv2.VideoCapture(str(orig_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[stage8.8] Could not open video: {orig_video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (dets[-1][1] + grow_frames + 1)
    limit = min(max_frames, total_frames) if max_frames is not None else total_frames
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    max_len = int(round(max_streak_seconds * fps))

    streaks = _link_streaks(dets, link_radius_px, grow_frames)

    tracks: List[dict] = []
    by_frame: Dict[int, List[tuple]] = defaultdict(list)
    for sid, st in enumerate(streaks):
        start, px, py = _sample_positions(st, grow_frames, limit - 1)
        n = len(px)
        tracks.append({
            'start': start,
            'sig': np.zeros(n, dtype=np.float32),
            'con': np.full(n, -1.0, dtype=np.float32),
            'x': np.zeros(n, dtype=np.int32),
            'y': np.zeros(n, dtype=np.int32),
        })
        for k in range(n):
            by_frame[start + k].append((sid, k, px[k], py[k]))

    last_needed = max(by_frame) if by_frame else -1
    fr = 0
    while fr <= last_needed:
        ok, frame = cap.read()
        if not ok:
            break
        todo = by_frame.get(fr)
        if todo:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for sid, k, x, y in todo:
                sig, con, bx, by = ring_contrast(gray, x, y)
                t = tracks[sid]
                t['sig'][k] = sig; t['con'][k] = con; t['x'][k] = bx; t['y'][k] = by
        _progress(fr + 1, last_needed + 1, 'stage8.8-grow'); fr += 1
    cap.release()

    taken_by_frame: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for _, f, cx, cy in dets:
        taken_by_frame[f].append((cx, cy))

    keep_idx = set()
    new_rows: List[dict] = []
    report: List[dict] = []
    reasons: Counter = Counter()
    for sid, st in enumerate(streaks):
        t = tracks[sid]
        start = t['start']
        seed_at = {d[1] - start: d[0] for d in st}
        for a, b in _flash_runs(t['con'], grow_contrast, max_frame_gap):
            seeds = sorted(k for k in seed_at if a <= k <= b)
            if not seeds:
                continue
            peak_con = float(t['con'][a:b + 1].max())
            width, base = _curve_shape(t['sig'], a, b, grow_frames)
            length = b - a + 1

            reason = ''
            if seed_contrast is not None and peak_con < float(seed_contrast):
                reason = 'weak'
            elif length > max_len:
                reason = 'too_long'
            elif width < int(min_width_frames):
                reason = 'too_narrow'
            elif base > float(max_base_frac):
                reason = 'flat'

            grown = 0
            if not reason:
                keep_idx.update(seed_at[k] for k in seeds)
                for k in range(a, b + 1):
                    if k in seed_at or t['con'][k] < grow_contrast:
                        continue
                    f = start + k
                    x = int(t['x'][k]); y = int(t['y'][k])
                    if any((x - ox) ** 2 + (y - oy) ** 2 <= dedupe_px ** 2 for ox, oy in taken_by_frame[f]):
                        continue
                    src = rows[seed_at[min(seeds, key=lambda s: abs(s - k))]]
                    row = dict(src)
                    row['frame'] = f
                    row['x'] = x if _is_center(src) else x - box_w // 2
                    row['y'] = y if _is_center(src) else y - box_h // 2
                    row['w'] = box_w; row['h'] = box_h
                    new_rows.append(row)
                    taken_by_frame[f].append((x, y))
                    grown += 1
            else:
                reasons[reason] += len(seeds)

            seed_dets = [d for d in st if a <= d[1] - start <= b]
            report.append({
                'flash_id': len(report),
                't_start': start + a,
                't_end': start + b,
                'x': int(round(float(np.median([d[2] for d in seed_dets])))),
                'y': int(round(float(np.median([d[3] for d in seed_dets])))),
                'n_seed': len(seeds),
                'n_grown': grown,
                'peak_contrast': round(peak_con, 2),
                'width_frames': width,
                'base_frac': round(base, 3),
                'length_frames': length,
                'kept': int(not reason),
                'reason': reason,
            })

    det_idx = {d[0] for d in dets}
    reasons['dim'] = len(det_idx) - len(keep_idx) - sum(reasons.values())

    new_rows.sort(key=lambda r: int(r['frame']))
    kept_rows = [r for i, r in enumerate(rows) if i not in det_idx or i in keep_idx] + new_rows
    _write_csv_rows(csv_path, kept_rows, cols)

    if report_csv_path is not None:
        report_cols = ['flash_id', 't_start', 't_end', 'x', 'y', 'n_seed', 'n_grown', 'peak_contrast',
                       'width_frames', 'base_frac', 'length_frames', 'kept', 'reason']
        _write_csv_rows(Path(report_csv_path), report, report_cols)

    before = len(det_idx)
    after = len(keep_idx) + len(new_rows)
    if verbose:
        dropped = '  '.join(f"{k} {v}" for k, v in sorted(reasons.items()) if v)
        print(f"[stage8.8] Fireflies before {before}  after {after}  "
              f"(kept {len(keep_idx)}, grown {len(new_rows)})  | flashes {len(report)}  kept {sum(r['kept'] for r in report)}")
        if dropped:
            print(f"[stage8.8] Seeds dropped: {dropped}")
    return (before, after, len(new_rows))
