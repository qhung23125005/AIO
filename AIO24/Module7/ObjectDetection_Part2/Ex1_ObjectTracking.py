import argparse
from collections import defaultdict
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from loguru import logger
import os

def load_config():
    return {
        'model_path': 'yolo11x.pt',
        'track_history_length': 120,
        'batch_size': 64,
        'line_thickness': 4,
        'track_color': (230, 230, 230)
    }

def initialize_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Safety check in case FPS is 0
    if fps == 0 or fps is None:
        fps = 30.0  # fallback to a default value

    video_name = os.path.basename(video_path)
    base_name = os.path.splitext(video_name)[0]

    # Ensure output directory exists
    output_dir = "run"
    os.makedirs(output_dir, exist_ok=True)

    output_path = f"{output_dir}/{base_name}_tracked.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Check if VideoWriter initialized correctly
    if not out.isOpened():
        raise IOError(f"Failed to create VideoWriter for: {output_path}")

    return cap, out, output_path

def update_track_history(
    track_history,
    last_seen,
    track_ids,
    frame_count,
    batch_size,
    frame_idx,
    history_length
):
    current_tracks = set(track_ids)
    for track_id in list(track_history.keys()):
        if track_id in current_tracks:
            last_seen[track_id] = frame_count - (batch_size - frame_idx - 1)
        elif frame_count - last_seen[track_id] > history_length:
            del track_history[track_id]
            del last_seen[track_id]

def draw_tracks(frame, boxes, track_ids, track_history, config):
    if not track_ids:
        return frame
    
    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))
        if len(track) > config['track_history_length']:
            track.pop(0)
        
        points = np.hstack(track).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            frame, 
            [points], 
            isClosed=False, 
            color=config['track_color'], 
            thickness=config['line_thickness']
        )
    return frame

def process_batch(model, batch_frames, track_history, last_seen, frame_count, config):
    results = model.track(
        batch_frames,
        persist = True,
        tracker = 'botsort.yaml',
        show = False,
        verbose = False,
        iou = 0.5
    )

    processed_frames = []
    for frame_idx, result in enumerate(results):
        boxes = result.boxes.xyxy.cpu()
        track_ids = result.boxes.id.cpu().tolist() if result.boxes.id is not None else []

        update_track_history(
            track_history,
            last_seen,
            track_ids,
            frame_count,
            config['batch_size'],
            frame_idx,
            config['track_history_length']
        )

        annotated_frame = result.plot(font_size=4, line_width=2)
        annotated_frame = draw_tracks(
            annotated_frame, 
            boxes, 
            track_ids, 
            track_history, 
            config
        ) # Draw lines that connect the historical points
        processed_frames.append(annotated_frame)
    return processed_frames

def mainn(video_path):
    CONFIG = load_config()
    model = YOLO(CONFIG.get('model_path', 'yolo11x.pt'))
    cap, out, output_path = initialize_video(video_path)
    track_history = defaultdict(lambda: [])
    last_seen = defaultdict(int)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames, desc="Processing frames", colour='green') as pbar:
        frame_count = 0
        batch_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            batch_frames.append(frame)
            if len(batch_frames) == CONFIG['batch_size'] or frame_count == total_frames:
                try:
                    processed_frames = process_batch(
                        model,
                        batch_frames,
                        track_history,
                        last_seen,
                        frame_count,
                        CONFIG
                    )
                    for processed_frame in processed_frames:
                        out.write(processed_frame)
                        pbar.update(1)
                    batch_frames = []
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    batch_frames = []
                    continue
    try:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        logger.info(f"Output saved to {output_path}")
    except Exception as e:
        logger.error(f"Error releasing resources: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=str, default = "data/vietnam.mp4")
    args = parser.parse_args()
    mainn(args.video_path)
    #main('data/vietnam.mp4')