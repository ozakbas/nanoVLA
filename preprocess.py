import os
import json
import pandas as pd
from moviepy.editor import VideoFileClip
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoModelForImageTextToText
import torch
from PIL import Image
import numpy as np
import cv2

from config import *
from helper_functions import list_recordings

assert torch.backends.mps.is_available(), "MPS not available. On Mac, enable Metal backend or switch device."
DTYPE = torch.bfloat16

if VLM_MODEL == "smolvlm":
    DTYPE = torch.float32

LOADED_MODELS = {}  

# ---------- frame sampling (used only for Qwen) ----------
def sample_frames_with_opencv(video_path: str, num_frames: int) -> list:
    """
    Samples frames uniformly from a video using OpenCV.
    Returns a list of PIL.Image objects in RGB.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise IOError(f"Video has zero frames: {video_path}")

    if total_frames < num_frames:
        print(f"Warning: Video has fewer frames ({total_frames}) than requested ({num_frames}). Using all frames.")
        num_frames = total_frames

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb_frame))
    cap.release()
    if not frames:
        raise IOError(f"No frames decoded: {video_path}")
    return frames

TOKENS_PER_FRAME = 32
MAX_PIXELS = TOKENS_PER_FRAME * 28 * 28

def _resize_to_max_pixels(img: Image.Image, max_pixels: int = MAX_PIXELS) -> Image.Image:
    w, h = img.size
    if w * h <= max_pixels:
        return img
    scale = (max_pixels / (w * h)) ** 0.5
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((nw, nh), Image.BICUBIC)


# ---------- Loaders ----------
def _load_qwen():
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=DTYPE,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True  
    ).eval()

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,  
    )
    return model, processor

def _load_smolvlm():
    model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=DTYPE,           
        attn_implementation="sdpa",
        trust_remote_code=True
    ).to(DEVICE)
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    try:
        if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "resample"):
            processor.image_processor.resample = Image.BICUBIC
    except Exception:
        pass
    return model, processor

MODEL_REGISTRY = {
    "qwen": _load_qwen,
    "smolvlm": _load_smolvlm,
}

def _canon(name: str) -> str:
    n = name.lower()
    if "qwen" in n: return "qwen"
    if "smol" in n: return "smolvlm"
    raise ValueError("Unsupported model_name. Use 'qwen' or 'smolvlm'.")

def _ensure_loaded(model_name: str):
    key = _canon(model_name)
    if key not in LOADED_MODELS:
        print(f"Loading '{key}' on MPS …")
        model, processor = MODEL_REGISTRY[key]()
        LOADED_MODELS[key] = (model, processor)
        print(f"Loaded '{key}' on {next(model.parameters()).device}.")
    return LOADED_MODELS[key]

# ---------- Postprocess ----------
def _extract_json(text: str) -> str:
    """
    Trim chat echo and return a clean JSON string if present.
    """
    import re
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return text.strip()
    try:
        obj = json.loads(m.group(0))
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return m.group(0).strip()

# ---------- Inference ----------
@torch.inference_mode()
def get_vlm_video_label(video_chunk_path: str, prompt: str, model_name: str, num_frames: int = 4) -> str:
    """
    Qwen: samples frames (multi-image input).
    SmolVLM: uses built-in mp4 path sampling (no OpenCV).
    """
    model, processor = _ensure_loaded(model_name)
    key = _canon(model_name)

    if key == "qwen":
        frames = sample_frames_with_opencv(video_chunk_path, num_frames)
        frames = [_resize_to_max_pixels(f) for f in frames]

        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": frames, "fps": 1.0},
                {"type": "text", "text": prompt + "\nReturn JSON only. No extra words."},
            ]
        }]

        text_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = processor(text=[text_prompt], videos=[frames], return_tensors="pt")
        for k, v in list(inputs.items()):
            if torch.is_tensor(v) and v.is_floating_point():
                inputs[k] = v.to(DTYPE)  


        out_ids = model.generate(
            **inputs,
            max_new_tokens=128,      
            do_sample=False,
            use_cache=True
        )

        new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
        out = processor.batch_decode(new_ids, skip_special_tokens=True)[0]
        print(f"Assistant's output: {out}")
        return _extract_json(out)


    if key == "smolvlm":
        print(video_chunk_path)

        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "path": video_chunk_path}, 
                {"type": "text", "text": prompt},
            ]
        }]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(DEVICE, dtype=DTYPE)
        

        input_token_len = inputs["input_ids"].shape[1]

        generated_ids = model.generate(
            **inputs, 
            do_sample=True,     
            temperature=0.7,    
            max_new_tokens=64
        )

        new_token_ids = generated_ids[:, input_token_len:]

        # Decode only the new tokens
        generated_text = processor.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
        )[0]
        
        print(f"Assistant's output: {generated_text}")
        return _extract_json(generated_text)

    raise ValueError(f"Unknown model: {model_name}")

# ---------- Split & Label pipeline ----------
def split_recordings():
    """
    Splits all recordings (CSV and MP4 files) into smaller chunks
    based on the duration specified in the config file.
    """
    print(">>> Starting the splitting process...")
    source_folder = RECORDINGS_FOLDER
    output_folder = SPLITTED_RECORDINGS_FOLDER
    split_duration = VIDEO_SPLIT_DURATION

    os.makedirs(output_folder, exist_ok=True)

    all_recordings = list_recordings()
    if not all_recordings:
        print("No recordings found to split.")
        return

    chunk_id_counter = 0
    unique_prefixes = sorted(list(set([f.split('_')[0] for f in all_recordings])))

    for prefix in unique_prefixes:
        csv_files = [f for f in all_recordings if f.startswith(f"{prefix}_") and f.endswith(".csv")]
        if not csv_files:
            continue

        latest_csv = max(csv_files, key=lambda f: os.path.getmtime(os.path.join(source_folder, f)))
        latest_mp4 = latest_csv.replace('.csv', '.mp4')
        csv_path = os.path.join(source_folder, latest_csv)
        video_path = os.path.join(source_folder, latest_mp4)

        if not os.path.exists(video_path):
            print(f"Warning: Video file not found for {latest_csv}, skipping this pair.")
            continue

        print(f"\nProcessing recording: {os.path.basename(csv_path)}")

        video = VideoFileClip(video_path)
        total_duration = video.duration

        start_time = 0
        while start_time < total_duration:
            end_time = min(start_time + split_duration, total_duration)

            # Output paths
            video_chunk_path = os.path.join(output_folder, f"{chunk_id_counter}.mp4")
            csv_chunk_path = os.path.join(output_folder, f"{chunk_id_counter}.csv")

            print(f"- Creating chunk {chunk_id_counter} ({start_time:.2f}s to {end_time:.2f}s)")

            subclip = video.subclip(start_time, end_time)
            subclip.write_videofile(video_chunk_path, codec="libx264", audio=False, logger=None)

            df = pd.read_csv(csv_path)
            time_mask = (df['Time'] >= start_time) & (df['Time'] < end_time)
            chunk_df = df[time_mask]
            if not chunk_df.empty:
                chunk_df.to_csv(csv_chunk_path, index=False)

            start_time += split_duration
            chunk_id_counter += 1

        video.close()

    print(f"\n>>> Splitting complete. Created {chunk_id_counter} chunks.")

def label_recordings():
    """
    Labels each video chunk in the splitted_recordings folder using a VLM
    and saves the results to a labels.csv file.
    """
    print("\n>>> Starting the labeling process...")
    source_folder = SPLITTED_RECORDINGS_FOLDER
    prompt = VLM_PROMPT

    video_chunks = sorted(
        [f for f in os.listdir(source_folder) if f.endswith('.mp4')],
        key=lambda x: int(os.path.splitext(x)[0])  
    )

    if not video_chunks:
        print("No video chunks found to label.")
        return

    results = []
    for video_file in video_chunks:
        video_path = os.path.join(source_folder, video_file)
        chunk_id = int(os.path.splitext(video_file)[0])


        label = get_vlm_video_label(video_path, prompt, VLM_MODEL)


        results.append({'id': chunk_id, 'label': label})

        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    output_csv_path = os.path.join(source_folder, 'labels.csv')
    labels_df = pd.DataFrame(results)
    labels_df.to_csv(output_csv_path, index=False)
    print(f"\n>>> Labeling complete. Results saved to {output_csv_path}")

if __name__ == "__main__":
    if not os.path.exists(SPLITTED_RECORDINGS_FOLDER):
        print("Splitted recordings folder not found. Running the splitting process first.")
        split_recordings()
    else:
        print("Splitted recordings folder already exists. Skipping the splitting process.")
    label_recordings()
