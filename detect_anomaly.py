import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(input_channels + hidden_channels,
                              4 * hidden_channels,
                              kernel_size,
                              padding=padding)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_output, 4, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTMAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.convlstm = ConvLSTMCell(64, 64, 3)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input_seq):
        b, t, c, h, w = input_seq.size()
        h_t, c_t = (torch.zeros(b, 64, h // 2, w // 2, device=input_seq.device),
                    torch.zeros(b, 64, h // 2, w // 2, device=input_seq.device))
        outputs = []

        for time in range(t):
            x = self.encoder(input_seq[:, time])
            h_t, c_t = self.convlstm(x, h_t, c_t)
            out = self.decoder(h_t)
            outputs.append(out.unsqueeze(1))

        return torch.cat(outputs, dim=1)

# === HELPER FUNCTION ===

def preprocess_frame(frame, size=(128, 128)):
    try:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, size)
        img = img.astype(np.float32) / 255.0
        return torch.tensor(img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return None

# === MAIN FUNCTION ===

@torch.no_grad()
def run_anomaly_detection(input_video_path, output_video_path):
    print(f"[INFO] Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvLSTMAutoencoder().to(device)
    model.load_state_dict(torch.load("conv_lstm_autoencoder_ucsd.pth", map_location=device))
    model.eval()
    print(f"[INFO] Model loaded.")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {input_video_path}")
        return

    frame_buffer = []
    sequence_length = 10
    output_frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream.")
            break

        frame_count += 1
        tensor = preprocess_frame(frame)
        if tensor is None:
            continue

        frame_buffer.append(tensor)

        if len(frame_buffer) == sequence_length:
            input_seq = torch.cat(frame_buffer, dim=0).unsqueeze(0).to(device)  # (1, 10, 1, 128, 128)
            print(f"[DEBUG] Inference on sequence: {input_seq.shape}")
            output_seq = model(input_seq)
            anomaly_map = torch.abs(input_seq - output_seq).mean(dim=2).squeeze().cpu().numpy()

            vis = frame.copy()
            heatmap = (anomaly_map[-1] * 255).astype(np.uint8)
            heatmap = cv2.resize(heatmap, (vis.shape[1], vis.shape[0]))
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            vis = cv2.addWeighted(vis, 0.6, heatmap, 0.4, 0)

            output_frames.append(vis)
            frame_buffer.pop(0)

    cap.release()

    print(f"[INFO] Total frames read: {frame_count}")
    print(f"[INFO] Total output frames generated: {len(output_frames)}")

    if output_frames:
        h, w, _ = output_frames[0].shape
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'avc1'), 25, (w, h))
        for f in output_frames:
            out.write(f)
        out.release()
        print(f"Output saved: {output_video_path} ({os.path.getsize(output_video_path)} bytes)")

        # === Optional: Re-encode with FFmpeg for browser support ===
        try:
            import subprocess
            temp_path = output_video_path.replace(".mp4", "_final.mp4")
            subprocess.run([
                "ffmpeg", "-y", "-i", output_video_path,
                "-c:v", "libx264", "-preset", "fast", "-movflags", "+faststart",
                temp_path
            ], check=True)
            os.replace(temp_path, output_video_path)
            print(f"🎥 Re-encoded for browser: {output_video_path}")
        except Exception as e:
            print(f"FFmpeg re-encoding failed: {e}")
    else:
        print("No output frames generated.")
