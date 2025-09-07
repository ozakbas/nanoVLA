import time
import os
import csv
import json
from datetime import datetime
import cv2
from servo_handler import ServoHandler
from video_handler import VideoHandler
import config as cfg
import threading

def _load_or_create_rest_positions(servo_handler, rest_pos_filepath):
    """
    If rest_positions.json exists and is valid, load & return it.
    Otherwise read current servo positions, save, and return the new map.
    """
    # Try to load existing
    if os.path.exists(rest_pos_filepath):
        try:
            with open(rest_pos_filepath, "r") as f:
                raw = json.load(f)
            rest_positions = {int(k): int(v) for k, v in raw.items()}
            print(f"📂 Loaded rest positions from {rest_pos_filepath}: {rest_positions}")
            return rest_positions
        except Exception as e:
            print(f"Failed to read/parse existing rest file ({e}).")

    # Build new
    rest_positions = {}
    for servo_id in cfg.SERVO_IDS:
        pos = servo_handler.read_position(servo_id)
        rest_positions[servo_id] = 0 if (pos is None or pos == -1) else int(pos)
    with open(rest_pos_filepath, "w") as f:
        json.dump(rest_positions, f)
    print(f"Created rest positions: {rest_positions}")
    print(f"Saved newrest positions to {rest_pos_filepath}")
    return rest_positions


def return_to_rest_position(servo_handler, rest_positions, steps: int = 50, step_delay: float = 0.03):
    """
    Move all servos to their REST absolute positions using only 'steps' linear interpolation.
    """
    print("\nReturning to the rest position")

    # Ensure torque is ON
    for sid in cfg.SERVO_IDS:
        servo_handler.set_torque(sid, True)

    # Read current positions
    current = []
    for sid in cfg.SERVO_IDS:
        pos = servo_handler.read_position(sid)
        current.append(0 if pos is None else (pos % 4096))

    # Shortest circular delta on 0..4095 ring
    def circ_delta(target, curr, max_val=4096):
        d = (target - curr) % max_val
        if d > max_val / 2:
            d -= max_val
        return d

    deltas = [circ_delta(rest_positions[sid] % 4096, curr) for sid, curr in zip(cfg.SERVO_IDS, current)]

    # Interpolate in 'steps'
    for i in range(1, steps + 1):
        alpha = i / steps
        targets = [ (curr + int(round(d * alpha))) % 4096 for curr, d in zip(current, deltas) ]
        for sid, pos in zip(cfg.SERVO_IDS, targets):
            servo_handler.move_servo(sid, pos)
        if step_delay > 0:
            time.sleep(step_delay)

    print("Reached rest positions smoothly.")


def record_movements(recording_id):
    """
    Records servo movements and video, continuing until the user presses Enter to stop.
    Uses existing rest_positions.json if available; otherwise creates it from current positions.
    """
    os.makedirs(cfg.RECORDINGS_FOLDER, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = cfg.MOVEMENT_FILENAME_TEMPLATE.format(recording_id, timestamp)
    video_filename = cfg.VIDEO_FILENAME_TEMPLATE.format(recording_id, timestamp)

    csv_filepath = os.path.join(cfg.RECORDINGS_FOLDER, csv_filename)
    video_filepath = os.path.join(cfg.RECORDINGS_FOLDER, video_filename)
    rest_pos_filepath = os.path.join(cfg.RECORDINGS_FOLDER, cfg.REST_POSITIONS_FILENAME)

    SAMPLE_DT = float(cfg.SAMPLING_INTERVAL)
    VIDEO_FPS = float(cfg.VIDEO_FPS)

    servo_handler = ServoHandler()
    video_handler = VideoHandler(video_filepath)

    try:
        servo_handler.connect()
        if hasattr(video_handler, "set_fps"):
            video_handler.set_fps(VIDEO_FPS)
        elif hasattr(video_handler, "fps"):
            video_handler.fps = VIDEO_FPS

        video_handler.start_recording()

        for servo_id in cfg.SERVO_IDS:
            try:
                servo_handler.set_midpoint_as_current_position(servo_id)
            except Exception:
                pass

        rest_positions = _load_or_create_rest_positions(servo_handler, rest_pos_filepath)
        return_to_rest_position(servo_handler, rest_positions)
        
        print("\nDisabling torque for manual movement...")
        for servo_id in cfg.SERVO_IDS:
            servo_handler.set_torque(servo_id, False)

        input("\nPress Enter to start recording...")
        print("\nRecording started... Press Enter again to stop.")

        stop_event = threading.Event()
        def stop_listener():
            """Waits for user input in a separate thread and sets an event."""
            input() # This will block until Enter is pressed
            stop_event.set()

        listener_thread = threading.Thread(target=stop_listener)
        listener_thread.start()

        t0 = time.monotonic()
        movement_data = []
        k = 0 # Step counter

        while not stop_event.is_set():
            target_time = t0 + k * SAMPLE_DT
            sleep_duration = target_time - time.monotonic()
            if sleep_duration > 0:
                time.sleep(sleep_duration)
            
            # Re-check the flag in case Enter was pressed during sleep
            if stop_event.is_set():
                break

            # Capture video frame exactly once per sample
            video_handler.capture_frame()

            # Read current servo positions
            current_positions = [servo_handler.read_position(sid) for sid in cfg.SERVO_IDS]

            # Prepare positions for saving (handle failed reads)
            absolute_positions_to_save = []
            for pos in current_positions:
                if pos is None or pos == -1:
                    absolute_positions_to_save.append("")
                else:
                    absolute_positions_to_save.append(pos)
            
            # Use deterministic timestamp based on sample rate
            t_rec = k * SAMPLE_DT
            movement_data.append([f"{t_rec:.4f}"] + absolute_positions_to_save)

            print(f"Time: {t_rec:5.2f}s | Abs: {absolute_positions_to_save}", end='\r')
            
            k += 1

        print("\nRecording stopped by user.")
        listener_thread.join()

        # Save data 
        with open(csv_filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time"] + [f"Servo_{sid}" for sid in cfg.SERVO_IDS])
            writer.writerows([row for row in movement_data if row])

        print(f"\nRecording complete. Data saved to '{csv_filepath}'.")
        print(f"Video saved to '{video_filepath}' at ~{VIDEO_FPS} FPS.")

    except Exception as e:
        print(f"\n An error occurred: {e}")
    finally:
        print("\nCleaning up...")
        return_to_rest_position(servo_handler, rest_positions)
        # Leave torque disabled so user can reposition if needed
        for servo_id in cfg.SERVO_IDS:
            servo_handler.set_torque(servo_id, False)
        servo_handler.disconnect()
        video_handler.stop_recording()

def sleep_until(t_rec):
    now = time.monotonic()
    target = t0 + t_rec
    dt = target - now
    if dt > 0:
        time.sleep(dt)


def list_recordings():
    """Lists all available .csv recordings in the recordings folder."""
    if not os.path.exists(cfg.RECORDINGS_FOLDER):
        print("Recordings folder not found.")
        return []
    files = [f for f in os.listdir(cfg.RECORDINGS_FOLDER) if f.endswith('.csv')]
    return files


def replay_movements(csv_filename):
    """
    Replays servo movements from a given CSV file.
    """
    csv_filepath = os.path.join(cfg.RECORDINGS_FOLDER, csv_filename)
    rest_pos_filepath = os.path.join(cfg.RECORDINGS_FOLDER, cfg.REST_POSITIONS_FILENAME)

    # load data
    try:
        with open(rest_pos_filepath, 'r') as f:
            rest_positions = {int(k): v for k, v in json.load(f).items()}

        with open(csv_filepath, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            movement_data = [row for row in reader]

    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure you have recorded first.")
        return

    servo_handler = ServoHandler()

    try:
        servo_handler.connect()
        for servo_id in cfg.SERVO_IDS:
            servo_handler.set_torque(servo_id, True)

        input("\nPress Enter to begin replay...")

        return_to_rest_position(servo_handler, rest_positions)

        print("\nStarting replay...")
        
        global t0
        t0 = time.monotonic()

        FRAME_TOL = cfg.SAMPLING_INTERVAL * 0.5

        for row in movement_data:
            try:
                recorded_time = float(row[0])
            except Exception:
                continue  # Skip bad row

            absolute_positions = []
            for s in row[1:]:
                s = (s or "").strip()
                if s == "":
                    absolute_positions.append(None)
                else:
                    try:
                        # Using float() first makes it robust to values like "123.0"
                        absolute_positions.append(int(float(s)))
                    except (ValueError, TypeError):
                        absolute_positions.append(None)

            # Precise timing: wait until this sample’s recorded timestamp
            sleep_until(recorded_time)

            # If we're late by more than half a sample, skip moving
            lag = (time.monotonic() - t0) - recorded_time
            if lag > (cfg.SAMPLING_INTERVAL + FRAME_TOL):
                continue
            
            print(f"t={recorded_time:5.2f}s  abs={absolute_positions}")
            for sid, pos in zip(cfg.SERVO_IDS, absolute_positions):
                if pos is not None:
                    servo_handler.move_servo(sid, pos)

        print("\nReplay complete.")
        return_to_rest_position(servo_handler, rest_positions)

    except (KeyboardInterrupt, Exception) as e:
        print(f"\nReplay stopped. Error: {e}")
    finally:
        print("\nCleaning up...")
        for servo_id in cfg.SERVO_IDS:
            servo_handler.set_torque(servo_id, False)
        servo_handler.disconnect()