import os
import requests
import time
import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np

# ----------- CONFIG ----------
IMAGE_DIR = "./coco_sample/val2017"
FOG_URL = "http://192.168.29.228:5000/upload"
CLOUD_URL = "https://1bcc-2409-40f0-1129-b35d-15d9-3526-c834-7b03.ngrok-free.app/upload"
MAX_WIDTH = 640
FOG_SIZE_LIMIT_KB = 400
MAX_RESOLUTION = 800

fog_times = []
cloud_times = []
fog_images = []
cloud_images = []
fog_bandwidth_kb = []
cloud_bandwidth_kb = []

# ----------- HELPER ----------
def compress_and_resize(image_path):
    img = Image.open(image_path).convert("RGB")
    scale = MAX_WIDTH / img.width
    new_size = (MAX_WIDTH, int(img.height * scale))
    img = img.resize(new_size)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=80)
    byte_data = buffer.getvalue()
    return byte_data, new_size

# ----------- MAIN LOOP ---------- 
for filename in os.listdir(IMAGE_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    filepath = os.path.join(IMAGE_DIR, filename)
    image_bytes, size = compress_and_resize(filepath)
    size_kb = len(image_bytes) / 1024
    width, height = size

    use_fog = size_kb <= FOG_SIZE_LIMIT_KB and width <= MAX_RESOLUTION and height <= MAX_RESOLUTION
    url = FOG_URL if use_fog else CLOUD_URL

    files = {'file': (filename, image_bytes, 'image/jpeg')}
    start = time.time()
    try:
        response = requests.post(url, files=files, timeout=30)
        end = time.time()
        elapsed = end - start

        if response.status_code == 200:
            if use_fog:
                fog_times.append(elapsed)
                fog_images.append(filename)
                fog_bandwidth_kb.append(size_kb)
            else:
                cloud_times.append(elapsed)
                cloud_images.append(filename)
                cloud_bandwidth_kb.append(size_kb)
            print(f"✅ {filename} -> {'FOG' if use_fog else 'CLOUD'} | Time: {elapsed:.3f}s")
        else:
            print(f"❌ {filename} -> Error {response.status_code}")

    except Exception as e:
        print(f"❌ {filename} -> Exception: {e}")

# ----------- METRICS ----------
avg_fog = np.mean(fog_times) if fog_times else 0
avg_cloud = np.mean(cloud_times) if cloud_times else 0
jitter_fog = np.std(fog_times) if fog_times else 0
jitter_cloud = np.std(cloud_times) if cloud_times else 0

bandwidth_fog_mb = np.sum(fog_bandwidth_kb) / 1024  # KB to MB
bandwidth_cloud_mb = np.sum(cloud_bandwidth_kb) / 1024

# ----------- BAR PLOTS ----------
labels = ['Fog', 'Cloud']
x = np.arange(len(labels))

response_times = [avg_fog, avg_cloud]
jitters = [jitter_fog, jitter_cloud]
bandwidths = [bandwidth_fog_mb, bandwidth_cloud_mb]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Avg Response Time
axs[0].bar(x, response_times, color=['blue', 'orange'])
axs[0].set_title('Avg Response Time')
axs[0].set_ylabel('Time (s)')
axs[0].set_xticks(x)
axs[0].set_xticklabels(labels)

# Jitter
axs[1].bar(x, jitters, color=['blue', 'orange'])
axs[1].set_title('Jitter (Std Dev)')
axs[1].set_ylabel('Time (s)')
axs[1].set_xticks(x)
axs[1].set_xticklabels(labels)

# Bandwidth Usage
axs[2].bar(x, bandwidths, color=['blue', 'orange'])
axs[2].set_title('Estimated Bandwidth Usage')
axs[2].set_ylabel('MB')
axs[2].set_xticks(x)
axs[2].set_xticklabels(labels)

plt.tight_layout()
plt.show()

# ----------- SUMMARY ----------
print("\n📈 Summary Metrics:")
print(f"Fog Images Processed: {len(fog_times)}")
print(f"Cloud Images Processed: {len(cloud_times)}")
print(f"Avg Fog Time: {avg_fog:.3f}s | Jitter: {jitter_fog:.3f}s | Bandwidth: {bandwidth_fog_mb:.2f} MB")
print(f"Avg Cloud Time: {avg_cloud:.3f}s | Jitter: {jitter_cloud:.3f}s | Bandwidth: {bandwidth_cloud_mb:.2f} MB")
