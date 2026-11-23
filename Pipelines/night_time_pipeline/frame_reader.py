#!/usr/bin/env python3
import queue
import threading

_DONE = object()


def read_ahead(items, depth: int = 1):
    """
    Runs the iterator `items` in a background thread and yields what it produces,
    with up to `depth` finished items waiting. An iterator that reuses its buffers
    needs depth + 2 of them: one being filled, `depth` waiting and one in use.
    """
    q = queue.Queue(maxsize=max(1, int(depth)))
    stop = threading.Event()

    def _put(item) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _work():
        try:
            for item in items:
                if not _put(item):
                    return
            _put(_DONE)
        except BaseException as exc:
            _put(exc)

    worker = threading.Thread(target=_work, daemon=True)
    worker.start()
    try:
        while True:
            item = q.get()
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        worker.join(timeout=10)


def frame_chunks(cap, chunk_frames: int, limit=None, buffers: int = 3):
    """
    Yields (index_of_first_frame, frames) for consecutive runs of up to chunk_frames
    frames, decoding into `buffers` sets of frames that are reused in turn.
    """
    ring = [[] for _ in range(buffers)]
    start = 0
    k = 0
    while True:
        frames = ring[k % buffers]
        n = 0
        while n < chunk_frames and (limit is None or start + n < limit):
            if n < len(frames):
                ok, frame = cap.read(frames[n])
            else:
                ok, frame = cap.read()
            if not ok:
                break
            if n < len(frames):
                frames[n] = frame
            else:
                frames.append(frame)
            n += 1
        if n == 0:
            return
        yield start, frames[:n]
        start += n
        k += 1
        if n < chunk_frames:
            return
