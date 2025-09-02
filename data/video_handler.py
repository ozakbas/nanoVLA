import cv2
from config import VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT, VIDEO_SOURCE, VIDEO_FPS

class VideoHandler:
    def __init__(self, filepath):
        self.filepath = filepath
        self.cap = None
        self.writer = None
        self.is_recording = False
        self.fps = None           # optional override via set_fps()
        self.last_frame = None    # buffer to ensure CFR (duplicate last good frame if needed)

    def set_fps(self, fps: float):
        """Optionally override output FPS (e.g., 1 / SAMPLING_INTERVAL)."""
        self.fps = float(fps)

    def start_recording(self):
        """Initializes the camera and starts recording."""
        self.cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam.")

        out_fps = self.fps if self.fps else float(VIDEO_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.filepath, fourcc, out_fps, (VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT)
        )
        if not self.writer.isOpened():
            raise IOError("Cannot open VideoWriter.")

        self.is_recording = True
        print(f"Started video recording @ {out_fps} FPS. Saving to {self.filepath}")

    def capture_frame(self):
        """
        Captures a single frame and writes it to the video file.
        Guarantees exactly one frame is written per call:
          - If camera read succeeds → write resized frame.
          - If it fails → duplicate the last valid frame.
        """
        if not self.is_recording or self.cap is None or self.writer is None:
            print(f"Something is wrong with the camera: self.is_recording {self.is_recording} self.cap {self.cap}  self.writer {self.writer}")
            return

        ret, frame = self.cap.read()
        if ret and frame is not None:
            resized_frame = cv2.resize(frame, (VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT))
            self.last_frame = resized_frame
            self.writer.write(resized_frame)
            cv2.imshow('Recording...', resized_frame)
            cv2.waitKey(1) # gui fix to show camera footage while recording
            return
        else:
            print(ret, frame)

        # Fallback: duplicate the last known good frame
        if self.last_frame is not None:
            self.writer.write(self.last_frame)
            cv2.imshow('Recording...', self.last_frame)
            cv2.waitKey(1)

    def stop_recording(self):
        """Stops recording and releases all resources."""
        if self.is_recording:
            self.is_recording = False
            if self.cap:
                self.cap.release()
                self.cap = None
            if self.writer:
                self.writer.release()
                self.writer = None
            cv2.destroyAllWindows()
            print("Video recording stopped.")
