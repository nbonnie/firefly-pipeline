#!/usr/bin/env python3
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from frame_reader import frame_chunks, read_ahead
from stage2_recenter import recenter_frame_rows
from stage4_cnn_filter import (load_classifier, patch_transform, row_center, label_patches,
                               _center_crop_clamped_bgr_to_pil)
from stage7_merge import merge_frame_rows
from stage8_gaussian_centroid import refine_frame_rows, save_frame_crops
from stage8_5_blob_area_filter import blob_area_of_row
from stage8_7_large_flash_bfs import expand_frame_rows, _orc

BASE_COLS = ['frame', 'x', 'y', 'w', 'h']
CNN_COLS = BASE_COLS + ['class', 'background_logit', 'firefly_logit', 'firefly_confidence']
CENTER_COLS = CNN_COLS + ['xy_semantics']

_BAR = 50
def _progress(i, total, tag=''):
    total = max(1, int(total or 1))
    i = min(i, total)
    frac = min(1.0, max(0.0, i / float(total)))
    bar  = '=' * int(frac * _BAR) + ' ' * (_BAR - int(frac * _BAR))
    sys.stdout.write(f'\r{tag} [{bar}] {int(frac*100):3d}%')
    sys.stdout.flush()
    if i >= total: sys.stdout.write('\n')

def _as_read_back(row: dict, cols: List[str]) -> dict:
    return {k: '' if row.get(k) is None else str(row.get(k, '')) for k in cols}

def _write_csv(path: Path, rows: List[dict], cols: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        wri = csv.DictWriter(f, fieldnames=cols)
        wri.writeheader()
        for r in rows:
            wri.writerow({k: r.get(k, '') for k in cols})

def _prune_small_blobs(frame, rows_f: List[dict], floor: int, area_threshold_px: int, counts: Dict[str, int]) -> List[dict]:
    kept = []
    for r in rows_f:
        if r.get('class') != 'firefly':
            kept.append(r)
            continue
        counts['before'] += 1
        try:
            if blob_area_of_row(frame, r, floor) < int(area_threshold_px):
                counts['deleted'] += 1
                continue
        except Exception:
            pass
        kept.append(r)
    return kept

def run_stages_2_to_8_7(
    *,
    orig_path: Path,
    csv_path: Path,
    snapshot_csv_path: Path,
    crop_dir: Path,
    after_stage4_csv_path: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Stages 2, 3, 4, 7, 8, 8.5, 8.7 and 8.5 again in one pass over the video. Every frame
    goes through the same per-frame code the separate stages use, in the same order, and
    the main CSV, the area snapshot and the crop folders come out the same.

    Frames are decoded in the background a chunk at a time; the classifier runs once per chunk.
    Parameters are read from the orchestrator, as in Stage 8.6 and 8.7.

    Returns: seconds spent per stage (pixel work only) plus 'frames' (waiting for decoded frames).
    """
    P = _orc()
    max_frames = P.MAX_FRAMES

    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        print("No detections to recenter.")
        return {}

    by_frame = defaultdict(list)
    max_frame_in_csv = 0
    for idx, r in enumerate(rows):
        f = int(r['frame'])
        by_frame[f].append((idx, int(r['x']), int(r['y']), int(r['w']), int(r['h'])))
        max_frame_in_csv = max(max_frame_in_csv, f)

    cap = cv2.VideoCapture(str(orig_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {orig_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    recenter_limit = min(max_frame_in_csv, total_frames)
    if max_frames is not None:
        recenter_limit = min(recenter_limit, max_frames)
    sparse = len(by_frame) / float(max(1, recenter_limit + 1)) <= 0.10

    def _stage2_reads(fr: int) -> bool:
        if sparse:
            return max_frames is None or fr < max_frames
        return fr < recenter_limit

    limit = max_frame_in_csv + 1
    if max_frames is not None:
        limit = min(limit, max_frames)

    model, device = load_classifier(P.CNN_MODEL_PATH, P.CNN_BACKBONE, print_load_status=P.PRINT_LOAD_STATUS,
                                    fail_if_weights_missing=P.FAIL_IF_WEIGHTS_MISSING)
    transform = patch_transform(P.IMAGENET_NORMALIZE)
    patch_batch_size = int(P.NUM_PATCHES_BATCH_SIZE or 0)
    floor = P.MIN_PIXEL_BRIGHTNESS_TO_BE_CONSIDERED_IN_AREA_CALCULATION

    crop_dir.mkdir(parents=True, exist_ok=True)
    out_root = P.ROOT / "stage8.7" / Path(orig_path).stem / "replacements"
    out_root.mkdir(parents=True, exist_ok=True)

    keep_mask = [True] * len(rows)
    seen = set()
    snapshot: List[tuple] = []
    after_stage4: List[dict] = []
    final_rows: List[dict] = []

    merged = dict(before=0, after=0)
    crops_saved = 0
    area_first = dict(before=0, deleted=0)
    area_second = dict(before=0, deleted=0)
    metrics = dict(frames_seen=0, candidates=0, replaced=0, dedup_groups=0,
                   old_rows_removed=0, new_rows_inserted=0, skipped_dim_seed=0)
    spent = defaultdict(float)

    t_wait = time.perf_counter()
    for start, frames in read_ahead(frame_chunks(cap, int(P.SINGLE_PASS_CHUNK_FRAMES), limit)):
        t0 = time.perf_counter()
        spent['frames'] += t0 - t_wait

        area_ok: Dict[int, List[dict]] = {}
        for k, frame in enumerate(frames):
            fr = start + k
            items = by_frame.get(fr)
            if not items:
                continue
            seen.add(fr)
            if _stage2_reads(fr):
                recenter_frame_rows(frame, items, rows, keep_mask, W, H, P.BRIGHT_MAX_THRESHOLD, P.SEED_CONTRAST_THRESHOLD)
            for idx, _x, _y, _w, _h in items:
                if not keep_mask[idx]:
                    continue
                r = rows[idx]
                w = int(r['w']); h = int(r['h'])
                if w * h < P.AREA_THRESHOLD_PX:
                    continue
                row = {'frame': str(fr), 'x': str(int(r['x'])), 'y': str(int(r['y'])), 'w': str(w), 'h': str(h)}
                snapshot.append((idx, row))
                area_ok.setdefault(fr, []).append(row)
        t1 = time.perf_counter()
        spent['02_recenter'] += t1 - t0

        labeled: List[dict] = []
        pending_tensors, pending_meta = [], []
        for k, frame in enumerate(frames):
            fr = start + k
            for row in area_ok.get(fr, []):
                center = row_center(row, False)
                if center is None:
                    continue
                cx, cy = center
                pil = _center_crop_clamped_bgr_to_pil(frame, cx, cy, P.CNN_PATCH_W, P.CNN_PATCH_H)
                if P.DEBUG_SAVE_PATCHES_DIR:
                    P.DEBUG_SAVE_PATCHES_DIR.mkdir(parents=True, exist_ok=True)
                    pil.save(P.DEBUG_SAVE_PATCHES_DIR / f"f{fr:06d}_cx{int(round(cx))}_cy{int(round(cy))}.png")
                pending_tensors.append(transform(pil))
                pending_meta.append((row, cx, cy))
                if patch_batch_size > 0 and len(pending_tensors) >= patch_batch_size:
                    labeled.extend(label_patches(model, device, pending_tensors, pending_meta,
                                                 P.CNN_CLASS_TO_KEEP, P.FIREFLY_CONF_THRESH))
                    pending_tensors.clear(); pending_meta.clear()
            if patch_batch_size <= 0 and pending_tensors:
                labeled.extend(label_patches(model, device, pending_tensors, pending_meta,
                                             P.CNN_CLASS_TO_KEEP, P.FIREFLY_CONF_THRESH))
                pending_tensors.clear(); pending_meta.clear()
        labeled.extend(label_patches(model, device, pending_tensors, pending_meta,
                                     P.CNN_CLASS_TO_KEEP, P.FIREFLY_CONF_THRESH))
        if P.DROP_BACKGROUND_ROWS:
            labeled = [r for r in labeled if r.get('class') == 'firefly']

        classified: Dict[int, List[dict]] = defaultdict(list)
        for r in labeled:
            r = _as_read_back(r, CNN_COLS)
            classified[int(r['frame'])].append(r)
            if after_stage4_csv_path is not None:
                after_stage4.append(r)
        t2 = time.perf_counter()
        spent['04_cnn'] += t2 - t1

        for k, frame in enumerate(frames):
            fr = start + k
            dets = classified.get(fr)
            if not dets:
                continue

            ta = time.perf_counter()
            kept = merge_frame_rows(frame, dets, W, H, P.STAGE7_DIST_THRESHOLD_PX)
            merged['before'] += len(dets); merged['after'] += len(kept)

            tb = time.perf_counter()
            refined: List[dict] = []
            refine_frame_rows(frame, fr, kept, P.STAGE8_PATCH_W, P.STAGE8_PATCH_H, P.STAGE8_GAUSSIAN_SIGMA, refined, [])
            refined = [_as_read_back(r, CENTER_COLS) for r in refined]
            crops_saved += save_frame_crops(frame, fr, [r for r in refined if r.get('class') == 'firefly'], True, crop_dir)

            tc = time.perf_counter()
            rows_f = _prune_small_blobs(frame, refined, floor, P.AREA_THRESHOLD_PX, area_first)

            td = time.perf_counter()
            if rows_f:
                rows_f = expand_frame_rows(
                    frame, fr, rows_f,
                    has_class=True, has_xy_sem=True,
                    neighbor_intensity_threshold=P.STAGE8_7_INTENSITY_THR,
                    dedupe_dist_px=P.STAGE8_7_DEDUPE_PX,
                    min_square_area_px=P.STAGE8_7_MIN_SQUARE_AREA_PX,
                    gaussian_sigma=P.STAGE8_7_GAUSSIAN_SIGMA,
                    out_root=out_root, metrics=metrics,
                )
                rows_f = [_as_read_back(r, CENTER_COLS) for r in rows_f]

            te = time.perf_counter()
            final_rows.extend(_prune_small_blobs(frame, rows_f, floor, P.AREA_THRESHOLD_PX, area_second))

            tf = time.perf_counter()
            spent['07_merge'] += tb - ta
            spent['08_gauss'] += tc - tb
            spent['08_5_blob_area'] += td - tc
            spent['08_7_large_flash_bfs'] += te - td
            spent['08_5_after_8_7'] += tf - te

        _progress(start + len(frames), limit, 'stages 2-8.7')
        t_wait = time.perf_counter()

    cap.release()

    for fr, items in by_frame.items():
        if fr in seen:
            continue
        for idx, _x, _y, _w, _h in items:
            r = rows[idx]
            w = int(r['w']); h = int(r['h'])
            if w * h >= P.AREA_THRESHOLD_PX:
                snapshot.append((idx, {'frame': str(fr), 'x': str(int(r['x'])), 'y': str(int(r['y'])), 'w': str(w), 'h': str(h)}))

    cols = BASE_COLS
    if any(keep_mask):
        snapshot.sort(key=lambda s: s[0])
        _write_csv(snapshot_csv_path, [row for _, row in snapshot], BASE_COLS)
        if snapshot:
            cols = CENTER_COLS if merged['after'] else CNN_COLS
    _write_csv(csv_path, final_rows, cols)
    if after_stage4_csv_path is not None:
        _write_csv(after_stage4_csv_path, after_stage4, CNN_COLS if snapshot else BASE_COLS)

    if verbose:
        print("\nStage-7 summary")
        print(f"  Boxes before: {merged['before']}")
        print(f"  Boxes after : {merged['after']}")
        print(f"  Deleted     : {merged['before'] - merged['after']}")
        print(f"Stage-8: saved {crops_saved} crops to {crop_dir}")
        for counts in (area_first, area_second):
            print(f"[stage8.5] Main CSV: fireflies before {counts['before']}  "
                  f"after {counts['before'] - counts['deleted']}  deleted {counts['deleted']}")
        print("\nStage 8.7 summary")
        print(f"  Frames processed       : {metrics['frames_seen']}")
        print(f"  Candidate replacements : {metrics['candidates']}")
        print(f"  Dedup groups           : {metrics['dedup_groups']}")
        print(f"  Replacements kept      : {metrics['replaced']}")
        print(f"  Old rows removed       : {metrics['old_rows_removed']}")
        print(f"  New rows inserted      : {metrics['new_rows_inserted']}")
        print(f"  Seeds below threshold  : {metrics['skipped_dim_seed']}")
        print(f"  Crops folder           : {out_root}")
        parts = '  '.join(f"{k} {v:.1f}s" for k, v in sorted(spent.items()))
        print(f"[single pass] rows {len(rows)} -> {len(final_rows)}  | {parts}")

    return dict(spent)
