#!/usr/bin/env python3
import numpy as np

WIN_HALF = 20
GUARD_HALF = 6
CORE_HALF = 5


def ring_contrast(gray: np.ndarray, cx: float, cy: float):
    H, W = gray.shape[:2]
    cx = max(0, min(int(round(cx)), W - 1))
    cy = max(0, min(int(round(cy)), H - 1))

    x0, x1 = max(0, cx - WIN_HALF), min(W, cx + WIN_HALF + 1)
    y0, y1 = max(0, cy - WIN_HALF), min(H, cy + WIN_HALF + 1)
    win = gray[y0:y1, x0:x1].astype(np.float32)

    ring = np.ones(win.shape, dtype=bool)
    ring[max(0, cy - GUARD_HALF - y0):cy + GUARD_HALF + 1 - y0,
         max(0, cx - GUARD_HALF - x0):cx + GUARD_HALF + 1 - x0] = False
    vals = win[ring]
    surround = float(np.median(vals))
    noise = 1.4826 * float(np.median(np.abs(vals - surround)))

    cx0 = max(0, cx - CORE_HALF - x0)
    cy0 = max(0, cy - CORE_HALF - y0)
    core = win[cy0:cy + CORE_HALF + 1 - y0, cx0:cx + CORE_HALF + 1 - x0]
    iy, ix = np.unravel_index(int(core.argmax()), core.shape)

    signal = float(core[iy, ix]) - surround
    return signal, signal / max(noise, 1.0), x0 + cx0 + int(ix), y0 + cy0 + int(iy)
