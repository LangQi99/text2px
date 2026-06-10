"""
Web UI for Text2Px - Generate 16x16 pixel art from text.
Uses only standard library (no Flask dependency).
"""
import os
import sys
import io
import base64
import json
import torch
import numpy as np
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.dit import Text2PxDiT
from model.diffusion import GaussianDiffusion
from model.tokenizer import CharTokenizer

model = None
diffusion = None
tokenizer = None
device = None

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
        .subtitle {
            color: #888;
            margin-bottom: 40px;
            font-size: 0.9rem;
        }
        .container {
            background: #16213e;
            border-radius: 16px;
            padding: 32px;
            width: 100%;
            max-width: 600px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        .input-group {
            display: flex;
            gap: 12px;
            margin-bottom: 24px;
        }
        input[type="text"] {
            flex: 1;
            padding: 14px 18px;
            border: 2px solid #2a3a5e;
            border-radius: 10px;
            background: #0f1629;
            color: #eee;
            font-size: 1rem;
            font-family: inherit;
            outline: none;
            transition: border-color 0.3s;
        }
        input[type="text"]:focus { border-color: #667eea; }
        button {
            padding: 14px 28px;
            border: none;
            border-radius: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 1rem;
            font-family: inherit;
            cursor: pointer;
            transition: transform 0.2s, opacity 0.2s;
        }
        button:hover { transform: translateY(-2px); opacity: 0.9; }
        button:active { transform: translateY(0); }
        button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .result {
            text-align: center;
            min-height: 200px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .result img {
            image-rendering: pixelated;
            image-rendering: crisp-edges;
            width: 256px;
            height: 256px;
            border: 3px solid #2a3a5e;
            border-radius: 8px;
            background: repeating-conic-gradient(#333 0% 25%, #222 0% 50%) 50% / 16px 16px;
        }
        .result .original {
            width: 64px;
            height: 64px;
            margin-top: 12px;
            border: 2px solid #2a3a5e;
            border-radius: 4px;
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
            padding: 6px 14px;
            background: #0f1629;
            border: 1px solid #2a3a5e;
            border-radius: 20px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .chip:hover { border-color: #667eea; background: #1a2440; }
        .footer { margin-top: 40px; color: #444; font-size: 0.75rem; }
        .footer a { color: #667eea; text-decoration: none; }
    </style>
</head>
<body>
    <h1>Text2Px</h1>
    <p class="subtitle">Generate 16x16 pixel art from text descriptions</p>
    <div class="container">
        <div class="input-group">
            <input type="text" id="prompt" placeholder="e.g. diamond sword, golden apple..."
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
                <span class="chip" onclick="tryExample(this)">ender pearl</span>
                <span class="chip" onclick="tryExample(this)">iron pickaxe</span>
                <span class="chip" onclick="tryExample(this)">red potion bottle</span>
                <span class="chip" onclick="tryExample(this)">bucket with water</span>
                <span class="chip" onclick="tryExample(this)">green emerald gem</span>
                <span class="chip" onclick="tryExample(this)">wooden bow</span>
            </div>
        </div>
    </div>
    <p class="footer">
        Powered by <a href="https://github.com/LangQi99/text2px">Text2Px DiT</a> |
        16x16 RGBA | Diffusion Transformer
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


def generate_image(prompt):
    tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    mask = torch.tensor([tokenizer.get_mask(prompt)], dtype=torch.bool).to(device)

    with torch.no_grad():
        samples = diffusion.sample_ddim(model, tokens, mask, image_size=16, channels=4, steps=40)

    img_tensor = (samples[0] + 1) / 2.0
    img_tensor = img_tensor.clamp(0, 1)
    img_array = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    img = Image.fromarray(img_array, mode='RGBA')

    buf_orig = io.BytesIO()
    img.save(buf_orig, format='PNG')
    img_b64_orig = base64.b64encode(buf_orig.getvalue()).decode()

    img_large = img.resize((256, 256), Image.NEAREST)
    buf_large = io.BytesIO()
    img_large.save(buf_large, format='PNG')
    img_b64_large = base64.b64encode(buf_large.getvalue()).decode()

    return {
        'image_original': img_b64_orig,
        'image_large': img_b64_large,
    }


def load_model(checkpoint_path):
    global model, diffusion, tokenizer, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    model_config = config['model'].copy()
    model_config['vocab_size'] = checkpoint['tokenizer_vocab_size']

    model = Text2PxDiT(model_config).to(device)
    state_key = 'ema_model_state_dict' if 'ema_model_state_dict' in checkpoint else 'model_state_dict'
    model.load_state_dict(checkpoint[state_key])
    model.eval()

    tokenizer = CharTokenizer(max_len=config['model']['max_text_len'])
    tokenizer_path = os.path.join(config['data']['dataset_dir'], 'tokenizer.json')
    tokenizer.load(tokenizer_path)

    diffusion = GaussianDiffusion(
        timesteps=config['diffusion']['timesteps'],
        beta_schedule=config['diffusion']['beta_schedule'],
    )

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")


if __name__ == '__main__':
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/final_model.pt'

    if not os.path.exists(checkpoint_path):
        alt_paths = ['checkpoints/latest.pt', 'checkpoints/checkpoint_epoch200.pt',
                     'checkpoints/checkpoint_epoch150.pt', 'checkpoints/checkpoint_epoch100.pt']
        for p in alt_paths:
            if os.path.exists(p):
                checkpoint_path = p
                break

    print(f"Loading checkpoint: {checkpoint_path}")
    load_model(checkpoint_path)

    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
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
