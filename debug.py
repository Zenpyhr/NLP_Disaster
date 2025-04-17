import dill as pickle
import os

# List all pickle files in your working directory
pickle_files = [f for f in os.listdir() if f.endswith('.pkl')]

for filename in pickle_files:
    print(f"\n📦 Checking file: {filename}")
    try:
        with open(filename, "rb") as f:
            index = 0
            while True:
                try:
                    item = pickle.load(f)
                    shape = getattr(item, "shape", None)
                    ndim = getattr(item, "ndim", None)
                    print(f"  ➤ Item {index}: type={type(item)}, shape={shape}, ndim={ndim}")
                    if ndim == 1:
                        print(f"  ❗ Problematic 1D array found in {filename} at item {index}")
                    index += 1
                except EOFError:
                    break
    except Exception as e:
        print(f"  ❌ Failed to read {filename}: {e}")
