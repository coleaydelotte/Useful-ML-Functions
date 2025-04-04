import kagglehub
import shutil
import os

path = kagglehub.dataset_download("rohanrao/xeno-canto-bird-recordings-extended-n-z")

print("Path to dataset files:", path)

os.makedirs("./data", exist_ok=True)
shutil.move(path, "./data")
print("Dataset installed into the ./data directory.")