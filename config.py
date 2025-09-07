# -- Servo Configuration --
SERIAL_PORT = "/dev/cu.usbmodem58760433931"  
BAUD_RATE = 1000000  
SERVO_IDS = [1, 2, 3, 4, 5, 6]  

# -- Feetech Servo Address Map --
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_POSITION = 42
ADDR_PRESENT_POSITION = 56
ADDR_PRESENT_SPEED = 60
ADDR_PRESENT_VOLTAGE = 62
ADDR_PRESENT_TEMPERATURE = 63
ADDR_GOAL_ACC = 41
ADDR_GOAL_SPEED = 46
ADDR_MIDPOINT_OFFSET = 20

# -- Recording Settings --
VIDEO_SPLIT_DURATION = 5 
SAMPLING_INTERVAL = 0.05 
VIDEO_FPS = round(1.0 / SAMPLING_INTERVAL)
RECORDINGS_FOLDER = "data/recordings" 
SPLITTED_RECORDINGS_FOLDER = "data/splitted_recordings"
REST_POSITIONS_FILENAME = "rest_positions.json"
MOVEMENT_FILENAME_TEMPLATE = "{}_{}.csv"
VIDEO_FILENAME_TEMPLATE = "{}_{}.mp4"

# -- VLM for Automatic Labeling --
VLM_MODEL = "smolvlm" # qwen or smolvlm
VLM_PROMPT = """Analyze the LEGO building action in the video. Respond with a JSON object containing three keys: 
1. 'action': A single verb for the action (e.g., 'pick', 'place', 'rotate', 'press').
2. 'piece_description': A description of the LEGO piece being moved (e.g., 'blue 2x4 brick').
3. 'destination_description': A description of where the piece is placed (e.g., 'next to the red 2x2 brick').
"""
# -- Video Settings --
VIDEO_SOURCE = 0  
VIDEO_FRAME_WIDTH = 640
VIDEO_FRAME_HEIGHT = 480


# -- Training Hyperparameters --
NUM_EPOCHS = 200                 
BATCH_SIZE = 128
LEARNING_RATE = 5e-5

# -- Model Architecture Hyperparameters --
CHUNK_SIZE = 20                 # Number of actions predicted in a chunk (k)
ACTION_DIM = 6                  # Dimension of the robot's action space 
HIDDEN_DIM = 512                # Main hidden dimension for the Transformer
LATENT_DIM = 32                 # Dimension of the VAE's latent variable z 
N_ENCODER_LAYERS = 4
N_DECODER_LAYERS = 7
N_HEADS = 8
DIM_FEEDFORWARD = 3200
DROPOUT = 0.1
BETA = 10.0              


RESNET_FEATURE_DIM = 512 # Output feature dim of the image backbone

import torch
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")