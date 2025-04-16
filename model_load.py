import os
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError # For more specific error handling

# Define the ModelConfig structure (or just use dictionaries)
class ModelConfig:
    def __init__(self, name, model_path, is_local, model_type):
        self.name = name
        self.model_path = model_path
        self.is_local = is_local
        self.model_type = model_type

# List of models to download (from your input)
models_to_download = [
    ModelConfig(
        name="Llama-3.2-3b-it",
        model_path="meta-llama/Llama-3.2-3B-Instruct",
        is_local=False,
        model_type="causal"
    ),
    ModelConfig(
        name="Llama-3.1-8b-it",
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        is_local=False,
        model_type="causal"
    ),
]

# --- Configuration ---
# Specify the base directory where you want to save the models
# Models will be saved in subdirectories like './models/Llama-3.2-3B-Instruct'
download_base_dir = "./downloaded_models"

# Optional: Set to True if you want to use symlinks to the cache instead of copying files.
# False = downloads files directly into the folder (uses more disk space if cache already exists)
# True/auto = uses symlinks if possible (saves disk space, standard behavior)
use_symlinks = 'auto' # or False

# Optional: Provide your HF token string here if you don't want to use CLI login.
# hf_token = "hf_YOUR_TOKEN_HERE" # Replace with your actual token if needed
hf_token = None # Set to None to rely on CLI login (recommended)

# --- Download Logic ---
print("--- Starting Model Download ---")
print(f"Models will be saved under: {os.path.abspath(download_base_dir)}")
print("\nIMPORTANT:")
print("1. Ensure you have accepted the license terms for BOTH models on the Hugging Face Hub.")
print("2. Ensure you are logged in via `huggingface-cli login` OR have provided a token in the script.")
print("-" * 30)

# Ensure the base directory exists
os.makedirs(download_base_dir, exist_ok=True)

for config in models_to_download:
    model_id = config.model_path
    # Create a specific subdirectory for this model for better organization
    # Uses the last part of the model_path as the folder name
    model_save_name = model_id.split('/')[-1]
    local_model_dir = os.path.join(download_base_dir, model_save_name)

    print(f"\nAttempting to download: {model_id}")
    print(f"Target local directory: {local_model_dir}")

    try:
        # Use snapshot_download to get the entire model repository
        snapshot_location = snapshot_download(
            repo_id=model_id,
            local_dir=local_model_dir,
            local_dir_use_symlinks=use_symlinks,
            token=hf_token, # Pass the token if provided, otherwise library finds it from login
            # resume_download=True, # Uncomment to attempt resuming interrupted downloads
            # ignore_patterns=["*.safetensors"], # Example: Uncomment to exclude certain file types
        )
        print(f"✅ Successfully downloaded {model_id} to {snapshot_location}")

    except HfHubHTTPError as e:
        # Catch specific HTTP errors which often relate to authentication/permissions
        print(f"❌ ERROR downloading {model_id}: {e}")
        if e.response.status_code == 401 or e.response.status_code == 403:
             print("   This often means you haven't accepted the license agreement on the Hugging Face Hub page for this model,")
             print("   or you are not correctly logged in (`huggingface-cli login`).")
        elif e.response.status_code == 404:
             print("   Model not found. Check the 'model_path' for typos.")
        else:
             print("   An HTTP error occurred. Check your network connection and Hugging Face status.")

    except Exception as e:
        # Catch any other unexpected errors
        print(f"❌ An unexpected error occurred while downloading {model_id}: {e}")
        print("   Check disk space, permissions, and network connectivity.")

    print("-" * 30)

print("--- Model Download Process Finished ---")