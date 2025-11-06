#!/usr/bin/env python3
"""
Web-based Training Monitor for Prometheus
Open in browser at: http://localhost:8888
"""

import os
import time
import subprocess
import json
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

def get_gpu_stats():
    """Get GPU utilization and memory usage"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            return {
                'gpu_util': int(parts[0]),
                'mem_used': int(parts[1]),
                'mem_total': int(parts[2]),
                'temp': int(parts[3]),
                'power': float(parts[4]) if len(parts) > 4 else 0
            }
    except:
        pass
    return None

def read_training_log():
    """Read the latest training progress from log file"""
    log_path = Path("C:/Users/Prithvi Putta/prometheus/train/training_log.txt")
    if not log_path.exists():
        return {'status': 'waiting', 'message': 'Waiting for training to start...'}

    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Check if training completed
        if 'Training completed successfully' in content or 'completed successfully' in content:
            return {
                'status': 'completed',
                'message': 'Training completed successfully!',
                'completion_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }

        # Check if training failed
        if 'Training failed' in content or 'Error' in content[-1000:]:
            return {
                'status': 'error',
                'message': 'Training encountered an error. Check logs.'
            }

        # Parse training progress
        lines = content.split('\n')
        latest_info = {
            'status': 'training',
            'epoch': None,
            'step': None,
            'total_steps': None,
            'loss': None,
            'lr': None,
            'message': 'Training in progress...'
        }

        # Look for download progress
        if 'Fetching' in content and 'Downloading' not in content:
            return {
                'status': 'downloading',
                'message': 'Downloading model files (~26GB)...'
            }

        if 'Loading' in content[-500:]:
            return {
                'status': 'loading',
                'message': 'Loading model and initializing training...'
            }

        # Parse actual training lines
        for line in reversed(lines[-100:]):
            if 'Epoch' in line and 'Step' in line and 'Loss' in line:
                try:
                    parts = line.split('|')
                    for part in parts:
                        part = part.strip()
                        if 'Epoch' in part and ':' in part:
                            latest_info['epoch'] = part.split(':')[1].strip()
                        elif 'Step' in part and ':' in part:
                            step_info = part.split(':')[1].strip()
                            if '/' in step_info:
                                current, total = step_info.split('/')
                                latest_info['step'] = current.strip()
                                latest_info['total_steps'] = total.strip()
                        elif 'Loss' in part and ':' in part:
                            latest_info['loss'] = part.split(':')[1].strip()
                        elif 'LR' in part and ':' in part:
                            latest_info['lr'] = part.split(':')[1].strip()

                    if latest_info['loss']:
                        break
                except:
                    continue

        return latest_info
    except Exception as e:
        return {'status': 'error', 'message': f'Error reading log: {str(e)}'}

class MonitorHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(get_html_page().encode())
        elif self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()

            gpu_stats = get_gpu_stats()
            train_info = read_training_log()

            data = {
                'gpu': gpu_stats,
                'training': train_info,
                'timestamp': time.time()
            }
            self.wfile.write(json.dumps(data).encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        # Suppress logging
        pass

def get_html_page():
    return '''<!DOCTYPE html>
<html>
<head>
    <title>Prometheus Training Monitor</title>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="5">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 36px;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .status-badge {
            display: inline-block;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: bold;
            margin-top: 10px;
        }
        .status-training { background: #4CAF50; }
        .status-downloading { background: #FF9800; }
        .status-loading { background: #2196F3; }
        .status-completed { background: #4CAF50; animation: pulse 1.5s infinite; }
        .status-error { background: #f44336; }
        .status-waiting { background: #9E9E9E; }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        .card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .card h2 {
            font-size: 24px;
            margin-bottom: 20px;
            border-bottom: 2px solid rgba(255,255,255,0.3);
            padding-bottom: 10px;
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
            font-size: 18px;
        }
        .stat-label {
            font-weight: bold;
            opacity: 0.9;
        }
        .stat-value {
            font-family: 'Courier New', monospace;
            background: rgba(0,0,0,0.2);
            padding: 5px 15px;
            border-radius: 5px;
        }
        .progress-bar {
            width: 100%;
            height: 30px;
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .completion-message {
            text-align: center;
            font-size: 48px;
            padding: 40px;
            animation: celebrate 2s infinite;
        }
        @keyframes celebrate {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            opacity: 0.8;
            font-size: 14px;
        }
    </style>
    <script>
        async function updateStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();

                // Update training status
                const training = data.training;
                const statusEl = document.getElementById('status');
                const messageEl = document.getElementById('message');

                statusEl.className = 'status-badge status-' + training.status;
                statusEl.textContent = training.status.toUpperCase();
                messageEl.textContent = training.message;

                // Update training info
                if (training.status === 'training') {
                    document.getElementById('training-card').style.display = 'block';
                    document.getElementById('epoch').textContent = training.epoch || 'N/A';
                    document.getElementById('step').textContent = training.step || 'N/A';
                    document.getElementById('total-steps').textContent = training.total_steps || 'N/A';
                    document.getElementById('loss').textContent = training.loss || 'N/A';
                    document.getElementById('lr').textContent = training.lr || 'N/A';

                    if (training.step && training.total_steps) {
                        const progress = (parseInt(training.step) / parseInt(training.total_steps)) * 100;
                        document.getElementById('progress-fill').style.width = progress + '%';
                        document.getElementById('progress-text').textContent = progress.toFixed(1) + '%';
                    }
                } else if (training.status === 'completed') {
                    document.getElementById('completion').style.display = 'block';
                    // Play notification sound if available
                    if (window.Notification && Notification.permission === "granted") {
                        new Notification("Training Complete!", {
                            body: "Your Prometheus model has finished training!",
                            icon: "✅"
                        });
                    }
                }

                // Update GPU info
                if (data.gpu) {
                    document.getElementById('gpu-card').style.display = 'block';
                    document.getElementById('gpu-util').textContent = data.gpu.gpu_util + '%';
                    document.getElementById('gpu-mem').textContent =
                        data.gpu.mem_used + ' / ' + data.gpu.mem_total + ' MB';
                    document.getElementById('gpu-temp').textContent = data.gpu.temp + '°C';

                    const memProgress = (data.gpu.mem_used / data.gpu.mem_total) * 100;
                    document.getElementById('mem-progress-fill').style.width = memProgress + '%';
                    document.getElementById('mem-progress-text').textContent = memProgress.toFixed(1) + '%';
                }

                // Update timestamp
                document.getElementById('last-update').textContent =
                    new Date().toLocaleTimeString();

            } catch (error) {
                console.error('Error fetching status:', error);
            }
        }

        // Request notification permission
        if (window.Notification) {
            Notification.requestPermission();
        }

        // Update every 5 seconds
        setInterval(updateStatus, 5000);
        updateStatus();
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Prometheus Training Monitor</h1>
            <div id="status" class="status-badge status-waiting">WAITING</div>
            <p style="margin-top: 10px;" id="message">Loading...</p>
        </div>

        <div id="completion" style="display:none;">
            <div class="card completion-message">
                🎉 Training Complete! 🎉
            </div>
        </div>

        <div class="grid">
            <div class="card" id="gpu-card" style="display:none;">
                <h2>⚡ GPU Status</h2>
                <div class="stat-row">
                    <span class="stat-label">Utilization:</span>
                    <span class="stat-value" id="gpu-util">0%</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Memory:</span>
                    <span class="stat-value" id="gpu-mem">0 / 0 MB</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="mem-progress-fill" style="width: 0%">
                        <span id="mem-progress-text">0%</span>
                    </div>
                </div>
                <div class="stat-row" style="margin-top: 15px;">
                    <span class="stat-label">Temperature:</span>
                    <span class="stat-value" id="gpu-temp">0°C</span>
                </div>
            </div>

            <div class="card" id="training-card" style="display:none;">
                <h2>📊 Training Progress</h2>
                <div class="stat-row">
                    <span class="stat-label">Epoch:</span>
                    <span class="stat-value" id="epoch">N/A</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Step:</span>
                    <span class="stat-value">
                        <span id="step">0</span> / <span id="total-steps">0</span>
                    </span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-fill" style="width: 0%">
                        <span id="progress-text">0%</span>
                    </div>
                </div>
                <div class="stat-row" style="margin-top: 15px;">
                    <span class="stat-label">Loss:</span>
                    <span class="stat-value" id="loss">N/A</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Learning Rate:</span>
                    <span class="stat-value" id="lr">N/A</span>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>Last updated: <span id="last-update">--:--:--</span></p>
            <p>Training logs: C:\\Users\\Prithvi Putta\\prometheus\\train\\training_log.txt</p>
            <p>Model output: C:\\Users\\Prithvi Putta\\prometheus\\lora_models\\</p>
        </div>
    </div>
</body>
</html>'''

def run_server():
    server_address = ('', 8888)
    httpd = HTTPServer(server_address, MonitorHandler)
    print("\n" + "="*70)
    print("Web Monitor Started!")
    print("="*70)
    print("\nOpen in your browser:")
    print("  -> http://localhost:8888")
    print("  -> http://127.0.0.1:8888")
    print("\nPress Ctrl+C to stop the monitor (training continues)")
    print("="*70 + "\n")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()
