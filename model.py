import os
import urllib.request
import torch
import segmentation_models_pytorch as smp

# Default direct download URL from GitHub Release (or set via environment variable / secrets)
DEFAULT_MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/MaheshBaja/camoflouge/releases/download/v1.0.0/best_model.pth"
)


def get_default_device() -> torch.device:
    """Returns CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def download_checkpoint(url: str = DEFAULT_MODEL_URL, destination: str = "best_model.pth") -> str:
    """Downloads model checkpoint from remote URL if not present locally."""
    if os.path.exists(destination):
        return destination

    alt_path = "best_model_final_one.pth"
    if os.path.exists(alt_path):
        return alt_path

    print(f"[*] Checkpoint not found locally. Downloading from {url}...")
    try:
        # Create request with browser User-Agent to avoid blocking
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req) as response, open(destination, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"[OK] Downloaded model weights to {destination}")
        return destination
    except Exception as e:
        if os.path.exists(destination):
            os.remove(destination)
        raise FileNotFoundError(
            f"Checkpoint file '{destination}' was not found locally, and automatic download failed: {e}.\n"
            f"Please upload 'best_model.pth' to GitHub Releases (tag v1.0.0) at:\n"
            f"https://github.com/MaheshBaja/camoflouge/releases/new"
        ) from e


def create_model() -> torch.nn.Module:
    """
    Instantiates the exact U-Net architecture with ResNet34 encoder.
    Matches training configuration:
      - Architecture: U-Net
      - Encoder: ResNet34
      - Input channels: 3
      - Output classes: 1 (binary segmentation)
    """
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # Weights loaded directly from checkpoint
        in_channels=3,
        classes=1,
    )
    return model


def load_model(checkpoint_path: str = "best_model.pth", device: torch.device = None) -> tuple[torch.nn.Module, torch.device, dict]:
    """
    Loads the trained model weights from the checkpoint.
    
    Handles:
      - Automatic remote download if running in cloud/Streamlit Cloud
      - Missing file fallbacks (best_model.pth vs best_model_final_one.pth)
      - Checkpoint dict extraction (model_state_dict)
      - DataParallel state_dict prefix cleanup ('module.')
      - Strict state_dict matching
      - Setting model to eval mode
      - Device mapping (GPU/CPU)
    
    Returns:
        (model, device, checkpoint_metadata)
    """
    if device is None:
        device = get_default_device()

    # Ensure checkpoint is available (checks local or downloads from release)
    checkpoint_path = download_checkpoint(DEFAULT_MODEL_URL, checkpoint_path)

    model = create_model()

    # Load checkpoint safely on specified device
    checkpoint = torch.load(checkpoint_path, map_location=device)

    metadata = {}
    if isinstance(checkpoint, dict):
        # Extract metadata if available
        for key in ["epoch", "best_iou", "history"]:
            if key in checkpoint:
                metadata[key] = checkpoint[key]

        # Extract model state dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Clean DataParallel 'module.' prefix if present
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v

    # Load weights with strict matching
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(device)
    model.eval()

    return model, device, metadata


if __name__ == "__main__":
    print("[*] Testing model loader...")
    try:
        model, device, meta = load_model("best_model.pth")
        print(f"[OK] Model loaded successfully on device: {device}")
        print(f"[OK] Checkpoint metadata: {meta}")
        
        # Test forward pass with dummy tensor
        dummy_input = torch.randn(1, 3, 352, 352).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"[OK] Forward pass successful. Output shape: {output.shape}")
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
