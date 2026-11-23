# Night Pipeline Configuration

All routine night-pipeline configuration is in:

```text
Pipelines/night_time_pipeline/pipeline_params.py
```

The orchestrator imports all uppercase values from this file. Gateway runs import the module in a subprocess and override selected values dynamically.

## Root and Inputs

```python
ROOT = Path("/mnt/Samsung_SSD_2TB/temp to delete/night time pipeline inference output data")
DIR_ORIG_VIDEOS = ROOT / "original videos"
DIR_BGS_VIDEOS = ROOT / "BS videos"
DIR_CSV = ROOT / "csv files"
GT_CSV_PATH = ROOT / "ground truth" / "gt.csv"
```

Direct-run layout:

```text
ROOT/
  original videos/
    video.mp4
  ground truth/
    gt.csv
```

Optional:

```text
ROOT/BS videos/<same_video_name>
```

## Cleanup

```python
RUN_PRE_RUN_CLEANUP = True
CLEANUP_KEEP_DIRS = ("ground truth", "original videos")
CLEANUP_GT_FILENAME = "gt.csv"
```

Night cleanup is more aggressive than day cleanup because it also removes old `ground truth/gt.csv` and may copy a root-level CSV into the ground-truth folder.

Use direct cleanup only on disposable roots.

## Stage Toggles

Default toggles:

```python
RUN_STAGE1 = True
RUN_STAGE2 = True
RUN_STAGE3 = True
RUN_STAGE4 = True
RUN_STAGE5 = True
RUN_STAGE6 = True
RUN_STAGE7 = True
RUN_STAGE8 = True
RUN_STAGE8_5 = True
RUN_STAGE8_6 = False
RUN_STAGE8_7 = True
RUN_STAGE8_8 = True
RUN_STAGE8_9 = True
RUN_STAGE8_5_AFTER_8_7 = True
RUN_STAGE9 = True
RUN_STAGE10 = True
RUN_STAGE11 = True
RUN_STAGE12 = True
RUN_STAGE14 = True
```

Run structure:

```python
PIPELINE_PHASE = 'all'
SINGLE_DECODE_PASS = True
SINGLE_PASS_CHUNK_FRAMES = 16
```

- `PIPELINE_PHASE`: `'detect'` runs cleanup and Stage 1 only, `'post'` runs everything after Stage 1 on the CSV a `'detect'` run left behind. The gateway sets these for its two night queues.
- `SINGLE_DECODE_PASS`: stages 2 to 8.7 share one pass over the video. Needs `RUN_STAGE8_6 = False` and stages 2, 3, 4, 7, 8, 8.5, 8.7 all on; otherwise each stage reads the video itself.
- `SINGLE_PASS_CHUNK_FRAMES`: frames decoded ahead and classified together in that pass. Three chunks are held in memory (about 25 MB per 4K frame).

For fast debugging, it is common to temporarily disable render/analysis stages:

- `RUN_STAGE5`
- `RUN_STAGE6`
- `RUN_STAGE10`
- `RUN_STAGE11`
- `RUN_STAGE12`
- `RUN_STAGE14`

Keep detection/centroid/repair stages on unless you are intentionally isolating one stage.

## Frame Cap

```python
MAX_FRAMES = None
```

Use an integer for short smoke tests.

The gateway sets full-frame processing by default unless `--max-frames` is passed or inferred.

## Stage 1 Variant

```python
STAGE1_VARIANT = "cucim"
STAGE8_6_STAGE1_IMPL = STAGE1_VARIANT
```

Variants:

- `cucim` - fastest intended path on a correctly configured CUDA/CuPy/cuCIM machine.
- `cc_cuda` - CUDA connected-components path.
- `cc_cpu` - CPU connected-components fallback.
- `blob` - OpenCV SimpleBlobDetector fallback.

Use `cc_cpu` for environment smoke tests when cuCIM is failing.

## cuCIM Detector Knobs

```python
CUCIM_DETECTOR = "log"
CUCIM_MIN_SIGMA = 0.75
CUCIM_MAX_SIGMA = 4.0
CUCIM_NUM_SIGMA = 10
CUCIM_THRESHOLD = 0.08
CUCIM_OVERLAP = 0.5
CUCIM_USE_CLAHE = True
CUCIM_BATCH_SIZE = 250
```

Tuning guidance:

- Lower `CUCIM_THRESHOLD` for more recall.
- Increase `CUCIM_MIN_AREA_PX` to remove tiny noise earlier.
- Reduce `CUCIM_BATCH_SIZE` if GPU memory or NVRTC issues appear.

## Connected-Component Knobs

```python
CC_MIN_AREA_PX = 1
CC_FIXED_THRESHOLD = 90
CC_USE_CLAHE = True
CC_USE_TOPHAT = True
CC_USE_DOG = True
```

Use these when `STAGE1_VARIANT` is `cc_cpu` or `cc_cuda`.

## Recenter and Area Filters

```python
BRIGHT_MAX_THRESHOLD = 50
SEED_CONTRAST_THRESHOLD = 30.0
AREA_THRESHOLD_PX = 6
MIN_PIXEL_BRIGHTNESS_TO_BE_CONSIDERED_IN_AREA_CALCULATION = 20
```

These affect Stage 2, Stage 3, Stage 8.5, and validation GT filtering.

Raising thresholds improves precision but can miss dim flashes.

`SEED_CONTRAST_THRESHOLD` is the Stage 2 local-contrast gate. It is relative to each candidate's own surroundings, so bright sky and sensor grain do not pass it the way they pass the absolute thresholds. `None` disables it.

## CNN Filter

```python
USE_CNN_FILTER = True
CNN_MODEL_PATH = Path(...)
CNN_BACKBONE = "resnet18"
CNN_CLASS_TO_KEEP = 1
CNN_PATCH_W = 10
CNN_PATCH_H = 10
NUM_PATCHES_BATCH_SIZE = 16384
FIREFLY_CONF_THRESH = 0.5
DROP_BACKGROUND_ROWS = False
FAIL_IF_WEIGHTS_MISSING = True
```

Default direct-run model:

```text
/mnt/Samsung_SSD_2TB/temp to delete/firefly_patch_training_local_run/tmp_day_night_combo_train_and_infer__20260305__163237/models/night/global_all_species.pt
```

Gateway override:

```bash
--night-cnn-model "/path/to/best.pt"
```

## Merge and Centroid

```python
STAGE7_DIST_THRESHOLD_PX = 20.0
STAGE8_PATCH_W = 10
STAGE8_PATCH_H = 10
STAGE8_GAUSSIAN_SIGMA = 0.0
```

Stage 7 merge happens before center semantics are finalized. Stage 8 rewrites the CSV to center semantics.

## Repair Stages

```python
STAGE8_6_RUNS = 1
STAGE8_6_DEDUPE_PX = 4.0
STAGE8_7_INTENSITY_THR = 70
STAGE8_7_DEDUPE_PX = 10.0
STAGE8_7_MIN_SQUARE_AREA_PX = 75
```

Stage 8.6 improves recall around masked existing detections. It is off by default: with the Stage 2 contrast gate and Stage 8.8 it took about two thirds of the run time on a 4K video and almost none of its additions survived Stage 8.8. Set `RUN_STAGE8_6 = True` to turn it back on.

Stage 8.7 repairs large flashes that were split into smaller shards.

## Streak Grow and Shape Filter

```python
STAGE8_8_GROW_CONTRAST_THRESHOLD = 10.0
STAGE8_8_LINK_RADIUS_PX = 12.0
STAGE8_8_GROW_FRAMES = 15
STAGE8_8_MAX_FRAME_GAP = 2
STAGE8_8_MIN_WIDTH_FRAMES = 2
STAGE8_8_MIN_FLICKER_FRAMES = 3
STAGE8_8_MAX_BASE_FRAC = 0.35
STAGE8_8_MAX_STREAK_SECONDS = 3.0
STAGE8_8_DEDUPE_PX = 4.0
```

Stage 8.8 grows each seed streak down to the grow contrast and then keeps only flashes with a flash-like intensity curve.

- Lower `STAGE8_8_GROW_CONTRAST_THRESHOLD` to follow flash tails further into the noise.
- `STAGE8_8_MIN_WIDTH_FRAMES = 2` removes single-frame blips such as hot pixels. Use `1` for species whose flashes last one frame.
- `STAGE8_8_MIN_FLICKER_FRAMES = 3` keeps flashes that flicker: they never hold half their peak for two frames in a row, but reach it in at least this many frames.
- Raise `STAGE8_8_MAX_STREAK_SECONDS` for species with long glows.
- `STAGE8_8_GROW_FRAMES` limits how far a tail is followed and how far apart two seeds of one flash may be.

## Validation

```python
GT_CSV_PATH = ROOT / "ground truth" / "gt.csv"
GT_T_OFFSET = 0
DIST_THRESHOLDS_PX = [10.0]
STAGE9_ONLY_FIREFLY_ROWS = True
STAGE9_GT_DEDUPE_DIST_PX = 4.0
```

Night Stage 9 expects GT already normalized to:

```text
x,y,t
```

Stage 8.9 can rewrite `x,y,w,h,frame` GT into this format before Stage 9.

## What the Gateway Overrides

For night subprocesses, the gateway can override:

- `ROOT`
- output directories
- `GT_CSV_PATH`
- `CNN_MODEL_PATH`
- `STAGE9_MODEL_PATH`
- `MAX_FRAMES`
- `GT_T_OFFSET`
- `RUN_PRE_RUN_CLEANUP`
- Stage 9/10/11/12/14 toggles when `--force-tests` is passed

