import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Beta
import pandas as pd
import numpy as np
import argparse
import torchvision.transforms as T

from config import (
    SPLITTED_RECORDINGS_FOLDER, CHUNK_SIZE, ACTION_DIM, HIDDEN_DIM, N_DECODER_LAYERS,
    N_HEADS, DIM_FEEDFORWARD, DROPOUT, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, DEVICE,
    SERVO_MAX
)
from preprocess import _load_smolvlm

DTYPE = torch.float32
IMAGE_RESIZE = 512
POSITION_STD = 0.05

class LegoDataset(Dataset):
    def __init__(self, folder=SPLITTED_RECORDINGS_FOLDER, stride=5):
        self.labels_df = pd.read_csv(os.path.join(folder, 'labels.csv'))
        self.folder = folder
        self.stride = stride
        self.transform = T.Compose([T.Resize((IMAGE_RESIZE, IMAGE_RESIZE)), T.ToTensor()])
        self.instructions = []
        for _, row in self.labels_df.iterrows():
            try:
                label = json.loads(row['label'])
                instr = f"{label['action']} the {label['piece_description']} to the {label['destination_description']}"
            except:
                instr = "perform lego building action"
            self.instructions.append(instr)
        self.lengths = []
        for chunk_id in self.labels_df['id']:
            csv_path = os.path.join(folder, f"{chunk_id}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                sequence_length = len(df)
                num_subs = max(1, (sequence_length - CHUNK_SIZE) // stride + 1)
                self.lengths.append(num_subs)
            else:
                self.lengths.append(1)

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        chunk_cum = np.cumsum([0] + self.lengths)
        chunk_id = np.searchsorted(chunk_cum, idx, side='right') - 1
        sub_idx = idx - chunk_cum[chunk_id]
        start_t = sub_idx * self.stride
        csv_path = os.path.join(self.folder, f"{self.labels_df.iloc[chunk_id]['id']}.csv")
        df = pd.read_csv(csv_path)
        servo_cols = [f'Servo_{i}' for i in range(1, 7)]
        positions = df[servo_cols].values.astype(np.float32)  # [T, 6]
        A_raw = positions[start_t:start_t + CHUNK_SIZE]
        if len(A_raw) < CHUNK_SIZE:
            pad_len = CHUNK_SIZE - len(A_raw)
            A_raw = np.vstack([A_raw, np.tile(A_raw[-1:], (pad_len, 1))])
        A = A_raw / SERVO_MAX 
        state = positions[start_t] / 4096.0  
        video_path = os.path.join(self.folder, f"{self.labels_df.iloc[chunk_id]['id']}.mp4")
        instruction = self.instructions[chunk_id]
        prompt = f"{instruction}"
        return {
            'video_path': video_path,
            'prompt': prompt,
            'state': state,
            'A': torch.from_numpy(A),
        }


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, memory):
        attn_out, _ = self.attn(x, memory, memory)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, K, D = x.shape
        mask = torch.triu(torch.ones(K, K, device=x.device), diagonal=1).bool()
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class ActionExpert(nn.Module):
    def __init__(self, vlm_hidden_dim, hidden_dim, action_dim, n_layers, n_heads, dim_ff, dropout):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, action_dim)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i % 2 == 0:
                self.layers.append(CrossAttentionLayer(hidden_dim, n_heads, dim_ff, dropout))
            else:
                self.layers.append(SelfAttentionLayer(hidden_dim, n_heads, dim_ff, dropout))
        self.ln_f = nn.LayerNorm(hidden_dim)

    def forward(self, action_emb, memory):
        x = action_emb
        for layer in self.layers:
            if isinstance(layer, CrossAttentionLayer):
                x = layer(x, memory)
            else:
                x = layer(x)
        x = self.ln_f(x)
        v = self.output_head(x)
        return v


class NanoVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm, self.processor = _load_smolvlm()
        self.vlm.eval()
        for param in self.vlm.parameters():
            param.requires_grad = False
        vlm_hidden_dim = self.vlm.config.text_config.hidden_size
        self.state_proj = nn.Linear(ACTION_DIM, vlm_hidden_dim)
        self.action_expert = ActionExpert(
            vlm_hidden_dim, HIDDEN_DIM, ACTION_DIM, N_DECODER_LAYERS,
            N_HEADS, DIM_FEEDFORWARD, DROPOUT
        )
        self.feature_proj = nn.Linear(vlm_hidden_dim, HIDDEN_DIM)


    @torch.no_grad()
    def get_vlm_features(self, video_path, prompt, state=None):
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "path": video_path},
                {"type": "text", "text": prompt},
            ]
        }]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(DEVICE)

        if state is not None:
            token_embeddings = self.vlm.get_input_embeddings()(inputs['input_ids'])
            state_tensor = torch.from_numpy(state).to(DEVICE, dtype=DTYPE).unsqueeze(0)
            state_embedding = self.state_proj(state_tensor).unsqueeze(1)
            combined_embeddings = torch.cat([token_embeddings, state_embedding.to(token_embeddings.dtype)], dim=1)
            inputs['inputs_embeds'] = combined_embeddings
            del inputs['input_ids']
        
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(DTYPE)

        outputs = self.vlm(**inputs, output_hidden_states=True, use_cache=False)
        N = len(outputs.hidden_states) // 3
        hidden = outputs.hidden_states[N].mean(dim=1)
        return hidden


    def forward(self, batch):
        video_paths = batch['video_path']
        prompts = batch['prompt']
        states = batch['state']
        As = batch['A']
        B = len(video_paths)
        features = []
        for i in range(B):
            o = self.get_vlm_features(video_paths[i], prompts[i], states[i])
            features.append(o)
        o_t = torch.cat(features, dim=0)
        o_proj = self.feature_proj(o_t).unsqueeze(1)
        As = As.to(DEVICE)
        tau_dist = Beta(1.0, 3.0)
        taus = tau_dist.sample((B,)).to(DEVICE)
        epsilon = torch.randn_like(As) * POSITION_STD
        A_tau = taus.view(B, 1, 1) * As + (1 - taus.view(B, 1, 1)) * epsilon
        A_tau_emb = self.action_expert.action_proj(A_tau)
        v_pred = self.action_expert(A_tau_emb, o_proj)
        u_target = As - epsilon
        loss = F.mse_loss(v_pred, u_target)
        return loss

def collate_fn(batch):
    return {
        'video_path': [item['video_path'] for item in batch],
        'prompt': [item['prompt'] for item in batch],
        'state': np.stack([item['state'] for item in batch]),
        'A': torch.stack([item['A'] for item in batch])
    }

def train():
    dataset = LegoDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    print(f"Dataset size: {len(dataset)} samples")
    model = NanoVLA().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95))
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * NUM_EPOCHS
    print(f"Total training steps: {total_steps}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps-100, eta_min=2.5e-6)
    model.train()
    for epoch in range(NUM_EPOCHS):
        for step, batch in enumerate(dataloader):
            if epoch == 0 and step < 100:
                lr = LEARNING_RATE * (step + 1) / 100
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if not (epoch == 0 and step < 100):
                scheduler.step()
            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
    torch.save(model.state_dict(), "nanovla.pth")
    print("Training complete. Model saved to nanovla.pth")

@torch.no_grad()
def predict(video_path, prompt, state=None, num_steps=10):
    model = NanoVLA().to(DEVICE)
    model.load_state_dict(torch.load("nanovla.pth"))
    model.eval()
    o = model.get_vlm_features(video_path, prompt, state)
    o_proj = model.feature_proj(o).unsqueeze(1)
    # Initialize with repeated normalized state (or uniform ~0.5 if no state)
    if state is not None:
        init_pos = torch.from_numpy(state).unsqueeze(0).repeat(CHUNK_SIZE, 1)  
    else:
        init_pos = torch.full((CHUNK_SIZE, ACTION_DIM), 0.5, dtype=DTYPE, device=DEVICE)  
    A_pred = init_pos.unsqueeze(0)  # [1, CHUNK_SIZE, 6]
    taus = torch.linspace(0.0, 1.0, num_steps + 1, device=DEVICE)[1:]  
    epsilon = torch.randn_like(A_pred) * POSITION_STD  
    for tau in taus:
        A_tau = tau * A_pred + (1 - tau) * epsilon
        A_tau_emb = model.action_expert.action_proj(A_tau)
        v = model.action_expert(A_tau_emb, o_proj)
        A_pred = A_tau + (1 - tau) * v 
        A_pred = torch.clamp(A_pred, 0.0, 1.0) 
    return A_pred.squeeze(0).cpu().numpy()  * SERVO_MAX


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate with NanoVLA")
    parser.add_argument("--mode", choices=["train", "predict", ], default="train")
    parser.add_argument("--video", type=str, help="Video path for prediction")
    parser.add_argument("--prompt", type=str, default="perform lego building action", help="Language prompt")
    parser.add_argument("--state", type=str, help="Current joint states (space-separated floats)")
    parser.add_argument("--steps", type=int, default=10, help="Number of sampling steps")
    args = parser.parse_args()
    if args.mode == "train":
        train()
    elif args.mode == "predict":
        state = np.array([float(x) for x in args.state.split()]) if args.state else None
        actions = predict(args.video, args.prompt, state, args.steps)
        print("Predicted action chunk:\n", actions)