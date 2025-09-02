import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..'
))
sys.path.append(PROJECT_ROOT)
import config as cfg
from helper_functions import list_recordings, replay_movements

t0 = None  


if __name__ == "__main__":
    recordings = list_recordings()
    if not recordings:
        print("No recordings found.")
    else:
        raw = input("\nEnter recording index (will play CSV starting with index + '_': ").strip()
        if not raw.isdigit():
            print("Invalid index input.")
        else:
            idx = int(raw)
            prefix = f"{idx}_"
            matches = [f for f in recordings if f.startswith(prefix)]
            if not matches:
                print(f"No CSVs found starting with '{prefix}'.")
            else:
                # if multiple timestamps exist for the same index, pick the newest by mtime
                latest_match = max(
                    matches,
                    key=lambda f: os.path.getmtime(os.path.join(cfg.RECORDINGS_FOLDER, f))
                )
                print(f"Replaying: {latest_match}")
                replay_movements(latest_match)
