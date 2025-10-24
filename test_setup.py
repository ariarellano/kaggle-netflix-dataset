import numpy as np
import pandas as pd
import sklearn
import torch
import tensorflow as tf

print("✓ NumPy version:", np.__version__)
print("✓ Pandas version:", pd.__version__)
print("✓ Scikit-learn version:", sklearn.__version__)
print("✓ PyTorch version:", torch.__version__)
print("✓ TensorFlow version:", tf.__version__)
print("✓ CUDA available:", torch.cuda.is_available())



import torch
import tensorflow as tf

print("="*50)
print("PyTorch GPU Info:")
print("="*50)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Number of GPUs:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.current_device())
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("GPU Memory:", round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2), "GB")
else:
    print("⚠️ No GPU detected by PyTorch")

print("\n" + "="*50)
print("TensorFlow GPU Info:")
print("="*50)
gpu_devices = tf.config.list_physical_devices('GPU')
print("Number of GPUs detected:", len(gpu_devices))

if len(gpu_devices) > 0:
    for i, gpu in enumerate(gpu_devices):
        print(f"GPU {i}: {gpu}")
        details = tf.config.experimental.get_device_details(gpu)
        print(f"  Device name: {details.get('device_name', 'Unknown')}")
else:
    print("⚠️ No GPU detected by TensorFlow")