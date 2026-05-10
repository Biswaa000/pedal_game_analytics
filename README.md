# Padel Game Analytics - Shot Classification System

This repository contains a clean computer vision prototype for padel match analysis. It detects players, rackets, and the ball; tracks player identities across frames; classifies contact events as `FOREHAND`, `BACKHAND`, or `SMASH/SERVE`; and exports JSON, CSV, and annotated video outputs.

The prototype is intentionally lightweight: it uses COCO-pretrained YOLOv8m for player/racket detection, Ultralytics ByteTrack for player identity tracking, and OpenCV motion/blob logic as a fallback when the small fast-moving ball is not detected by YOLO.

## Pipeline

```text
                   +------------------+
                   |  input video     |
                   +---------+--------+
                             |
                             v
                   +------------------+
                   | detect.py        |
                   | YOLO + ball blob |
                   +---------+--------+
                             |
                             v
                   +------------------+
                   | track.py         |
                   | ByteTrack +      |
                   | racket matching  |
                   +---------+--------+
                             |
                             v
                   +------------------+
                   | classify.py      |
                   | contact rules    |
                   +---------+--------+
                             |
                             v
                   +------------------+
                   | pipeline.py      |
                   | JSON/CSV/video   |
                   +------------------+
```

## Setup

```bash
git clone <repo-url>
cd padel-shot-classifier
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the YOLOv8m weights and place them in `models/`:

```bash
python - <<'PY'
from ultralytics import YOLO
YOLO("yolov8m.pt")
PY
mv yolov8m.pt models/yolov8m.pt 2>/dev/null || true
```

If `models/yolov8m.pt` is not present, Ultralytics will still attempt to resolve `yolov8m.pt` automatically when the pipeline runs.

## Usage

```bash
python src/pipeline.py --video data/match.mp4
```

For the videos currently present in this repo:

```bash
python src/pipeline.py --video data/input_sample_video.mp4
python src/pipeline.py --video data/infernce_sample_video.mp4
```

Optional arguments:

```bash
python src/pipeline.py --video data/input_sample_video.mp4 --output-dir outputs --model models/yolov8m.pt
python src/pipeline.py --video data/input_sample_video.mp4 --device cuda:0
```

By default, `--device auto` checks `torch.cuda.is_available()`. If CUDA is available it runs YOLO on `cuda:0`; otherwise it falls back to `cpu`. Use `--device cuda:0` only after `nvidia-smi` works and PyTorch reports `torch.cuda.is_available() == True`.

## Docker

Build the image:

```bash
docker build -t padel-shot-classifier:latest .
```

Run the pipeline with local `data/`, `models/`, and `outputs/` mounted into the container:

```bash
docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/outputs:/app/outputs" \
  padel-shot-classifier:latest \
  --video data/match_1.mp4 --output-dir outputs
```

with GPU
```bash
docker run --rm --gpus all \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/outputs:/app/outputs" \
  padel-shot-classifier:latest \
  --video data/match_1.mp4 --output-dir outputs
```

Push to Docker Hub:

```bash
docker login
docker tag padel-shot-classifier:latest <dockerhub-username>/padel-shot-classifier:latest
docker push <dockerhub-username>/padel-shot-classifier:latest
```

The Docker image does not include local videos, model weights, or generated outputs. Mount those folders at runtime as shown above.

## Methodology

Stage 1, detection: `src/detect.py` loads YOLOv8m and runs inference on every frame. It extracts COCO `person` detections as players, `tennis racket` detections as rackets, and `sports ball` detections as the ball. If YOLO misses the ball, the module applies OpenCV MOG2 background subtraction, morphology, and `SimpleBlobDetector` to propose a small moving ball candidate.

Stage 2, tracking: `src/track.py` wraps Ultralytics tracking with `tracker="bytetrack.yaml"` to assign stable player IDs. Rackets are associated to the nearest plausible tracked player by IoU or centroid distance, and the ball state is merged from the YOLO/background-subtraction detection pass.

Stage 3, classification: `src/classify.py` detects likely ball-racket contact when the ball centroid is close to a racket centroid. It then uses racket position relative to the player centroid and a short racket velocity estimate over nearby frames to classify the shot. The rules separate overhead `SMASH/SERVE`, lateral `FOREHAND`, lateral `BACKHAND`, and `UNKNOWN`. A 15-frame per-player cooldown prevents duplicate contact events from being counted repeatedly.

Stage 4, orchestration and outputs: `src/pipeline.py` wires the modules together, writes `outputs/shots.json` and `outputs/shots.csv`, renders `outputs/annotated.mp4`, and prints a summary table to stdout. The annotated video draws green player boxes, blue racket boxes, a red ball marker, contact labels, and a running shot counter HUD.

## Challenges

Ball detection is the hardest part of this prototype because the padel ball is small, fast, often motion-blurred, and frequently occluded by players, glass, or racket motion. COCO-pretrained YOLO can detect generic sports balls, but it is not specialized for padel footage, camera angles, or ball scale.

Shot classification is also limited by the lack of pose labels and handedness information. The current geometric rules use racket position relative to the player box, which is useful for a prototype but can confuse forehand/backhand when players face opposite camera directions or rotate during play. Occlusion and overlapping players can further affect both player tracking and racket-player assignment.

## Future Improvements

- Add MediaPipe pose or another keypoint model to infer shoulders, elbows, wrists, stance, and handedness.
- Use TrackNet or a padel-specific ball detector for more reliable ball trajectories.
- Fine-tune YOLO on padel-specific player, racket, and ball annotations.
- Train a sequence model such as an LSTM or temporal transformer over pose, racket, and ball trajectories.
- Add court calibration to reason about player side, bounce location, and shot direction.

## Output Format

`outputs/shots.json` is a list of shot event dictionaries:

```json
[
  {
    "frame_idx": 342,
    "timestamp_sec": 11.4,
    "player_id": 1,
    "shot_type": "FOREHAND",
    "confidence": 0.82
  }
]
```

`outputs/shots.csv` contains the same fields as columns:

```text
frame_idx,timestamp_sec,player_id,shot_type,confidence
342,11.4,1,FOREHAND,0.82
```

The annotated render is written to `outputs/annotated.mp4`.
