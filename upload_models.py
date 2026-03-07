from huggingface_hub import HfApi
import os

api = HfApi()

# Login first - get token from huggingface.co/settings/tokens
repo_id = "RAyush257/mediscan-models"

files = [
    "models/cxr_model.h5",
    "models/malaria_model.h5", 
    "models/ocular_model.h5",
    "models/brain_model.h5",
    "models/cxr_classes.json",
    "models/malaria_classes.json",
    "models/ocular_classes.json",
    "models/brain_classes.json",
]

for filepath in files:
    filename = os.path.basename(filepath)
    print(f"Uploading {filename}...")
    api.upload_file(
        path_or_fileobj=filepath,
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model"
    )
    print(f"✅ {filename} uploaded!")

print("All models uploaded!")