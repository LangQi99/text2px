"""
Web UI for Text2Px - Generate 16x16 pixel art from text.
Uses the reference DiT pipeline with BPE tokenizer and CFG.
"""
import io
import sys
import base64
import json
import torch
import numpy as np
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.npy_dataset import denormalize_rgba
from diffusion import create_diffusion
from model.reference_dit import create_reference_dit

model = None
diffusion = None
tokenizer = None
device = None
metadata = None

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text2Px - Pixel Art Generator</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Courier New', monospace;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px 20px;
        }
        h1 {
            font-size: 2.5rem;
            margin-bottom: 8px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle { color: #888; margin-bottom: 40px; font-size: 0.9rem; }
        .container {
            background: #16213e;
            border-radius: 16px;
            padding: 32px;
            width: 100%;
            max-width: 600px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        .input-group { display: flex; gap: 12px; margin-bottom: 24px; }
        input[type="text"] {
            flex: 1; padding: 14px 18px;
            border: 2px solid #2a3a5e; border-radius: 10px;
            background: #0f1629; color: #eee;
            font-size: 1rem; font-family: inherit; outline: none;
            transition: border-color 0.3s;
        }
        input[type="text"]:focus { border-color: #667eea; }
        button {
            padding: 14px 28px; border: none; border-radius: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; font-size: 1rem; font-family: inherit;
            cursor: pointer; transition: transform 0.2s, opacity 0.2s;
        }
        button:hover { transform: translateY(-2px); opacity: 0.9; }
        button:active { transform: translateY(0); }
        button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .result {
            text-align: center; min-height: 200px;
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
        }
        .result img {
            image-rendering: pixelated; image-rendering: crisp-edges;
            width: 256px; height: 256px;
            border: 3px solid #2a3a5e; border-radius: 8px;
            background: repeating-conic-gradient(#333 0% 25%, #222 0% 50%) 50% / 16px 16px;
        }
        .result .original {
            width: 64px; height: 64px; margin-top: 12px;
            border: 2px solid #2a3a5e; border-radius: 4px;
        }
        .loading { color: #667eea; font-size: 1.1rem; }
        .loading::after { content: ''; animation: dots 1.5s steps(4) infinite; }
        @keyframes dots {
            0% { content: ''; } 25% { content: '.'; }
            50% { content: '..'; } 75% { content: '...'; }
        }
        .error { color: #ff6b6b; }
        .prompt-label { color: #888; font-size: 0.85rem; margin-top: 12px; }
        .examples { margin-top: 20px; padding-top: 20px; border-top: 1px solid #2a3a5e; }
        .examples h3 { font-size: 0.85rem; color: #666; margin-bottom: 10px; }
        .example-chips { display: flex; flex-wrap: wrap; gap: 8px; }
        .chip {
            padding: 6px 14px; background: #0f1629;
            border: 1px solid #2a3a5e; border-radius: 20px;
            font-size: 0.8rem; cursor: pointer; transition: all 0.2s;
        }
        .chip:hover { border-color: #667eea; background: #1a2440; }
        .footer { margin-top: 40px; color: #444; font-size: 0.75rem; }
        .footer a { color: #667eea; text-decoration: none; }
    </style>
</head>
<body>
    <h1>Text2Px</h1>
    <p class="subtitle">Generate 16x16 pixel art from text (DiT + Diffusion + CFG)</p>
    <div class="container">
        <div class="input-group">
            <input type="text" id="prompt" placeholder="e.g. diamond sword, tool box, shulker box..."
                   onkeydown="if(event.key==='Enter')generate()">
            <button id="genBtn" onclick="generate()">Generate</button>
        </div>
        <div class="result" id="result">
            <p style="color: #555;">Enter a description and click Generate</p>
        </div>
        <div class="examples">
            <h3>Try these:</h3>
            <div class="example-chips">
                <span class="chip" onclick="tryExample(this)">diamond sword</span>
                <span class="chip" onclick="tryExample(this)">golden apple</span>
                <span class="chip" onclick="tryExample(this)">tool box</span>
                <span class="chip" onclick="tryExample(this)">iron pickaxe</span>
                <span class="chip" onclick="tryExample(this)">treasure box</span>
                <span class="chip" onclick="tryExample(this)">shulker box</span>
                <span class="chip" onclick="tryExample(this)">emerald</span>
                <span class="chip" onclick="tryExample(this)">fire book</span>
            </div>
        </div>
    </div>
    <p class="footer">
        Powered by <a href="https://github.com/LangQi99/text2px">Text2Px DiT</a> |
        16x16 RGBA | Diffusion Transformer + CFG
    </p>
    <script>
        function tryExample(el) {
            document.getElementById('prompt').value = el.textContent;
            generate();
        }
        async function generate() {
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) return;
            const btn = document.getElementById('genBtn');
            const result = document.getElementById('result');
            btn.disabled = true;
            result.innerHTML = '<p class="loading">Generating</p>';
            try {
                const resp = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt: prompt})
                });
                const data = await resp.json();
                if (data.error) {
                    result.innerHTML = '<p class="error">' + data.error + '</p>';
                } else {
                    result.innerHTML =
                        '<img src="data:image/png;base64,' + data.image_large + '" alt="Generated pixel art">' +
                        '<img class="original" src="data:image/png;base64,' + data.image_original + '" alt="Original 16x16" title="Original 16x16">' +
                        '<p class="prompt-label">"' + prompt + '"</p>';
                }
            } catch (e) {
                result.innerHTML = '<p class="error">Error: ' + e.message + '</p>';
            }
            btn.disabled = false;
        }
    </script>
</body>
</html>"""


def encode_prompt(tok, prompt, length):
    encoded = tok.encode(prompt)
    bos = tok.token_to_id("<|bos|>")
    eos = tok.token_to_id("<|eos|>")
    pad = tok.token_to_id("<|pad|>")
    ids = [bos] + encoded.ids[: length - 2] + [eos]
    return ids + [pad] * (length - len(ids))


class Text2PxHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/generate':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body)
            prompt = data.get('prompt', '').strip()
            if not prompt:
                self._json_response({'error': 'Please enter a text description'})
                return
            try:
                result = generate_image(prompt)
                self._json_response(result)
            except Exception as e:
                self._json_response({'error': str(e)})
        else:
            self.send_response(404)
            self.end_headers()

    def _json_response(self, data):
        response = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0]}")


def generate_image(prompt, cfg_scale=3.0):
    token_length = int(metadata.get("token_length", 8))
    mean = metadata["rgba_mean"]
    std = metadata["rgba_std"]

    token_ids = torch.tensor(
        [encode_prompt(tokenizer, prompt, token_length)],
        dtype=torch.long, device=device,
    )
    noise = torch.randn(1, 4, 16, 16, device=device)
    z = torch.cat([noise, noise], dim=0)
    token_cfg = torch.cat([token_ids, torch.zeros_like(token_ids)], dim=0)
    model_kwargs = dict(token_ids=token_cfg, cfg_scale=cfg_scale)

    with torch.no_grad():
        samples = diffusion.ddim_sample_loop(
            model.forward_with_cfg,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device,
            eta=0.0,
        )
    samples, _ = samples.chunk(2, dim=0)
    arr = denormalize_rgba(samples, mean, std)[0]
    img = Image.fromarray(arr, mode='RGBA')

    buf_orig = io.BytesIO()
    img.save(buf_orig, format='PNG')
    img_b64_orig = base64.b64encode(buf_orig.getvalue()).decode()

    img_large = img.resize((256, 256), Image.NEAREST)
    buf_large = io.BytesIO()
    img_large.save(buf_large, format='PNG')
    img_b64_large = base64.b64encode(buf_large.getvalue()).decode()

    return {'image_original': img_b64_orig, 'image_large': img_b64_large}


def load_model(checkpoint_path, tokenizer_path):
    global model, diffusion, tokenizer, device, metadata

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metadata = ckpt.get("metadata", {})
    model_name = metadata.get("model", "T2P-DiT-tiny")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    model = create_reference_dit(
        model_name,
        vocab_size=metadata.get("vocab_size", tokenizer.get_vocab_size()),
        token_seq_len=int(metadata.get("token_length", 8)),
        num_timesteps=int(metadata.get("num_timestep", 300)),
    ).to(device)
    key = "ema" if "ema" in ckpt else "model"
    model.load_state_dict(ckpt[key])
    model.eval()

    diffusion = create_diffusion(str(metadata.get("num_timestep", 300)))
    print(f"Model loaded: {model_name}, {sum(p.numel() for p in model.parameters()):,} parameters")


if __name__ == '__main__':
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/full_reference/final-model.pt'
    tokenizer_path = sys.argv[2] if len(sys.argv) > 2 else 'ref_artifacts/token-7524n.json'

    print(f"Loading checkpoint: {checkpoint_path}")
    load_model(checkpoint_path, tokenizer_path)

    port = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    server = HTTPServer(('0.0.0.0', port), Text2PxHandler)
    print(f"\n{'='*50}")
    print(f"  Text2Px Web UI running at http://0.0.0.0:{port}")
    print(f"  Open in browser to generate pixel art!")
    print(f"{'='*50}\n")
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
