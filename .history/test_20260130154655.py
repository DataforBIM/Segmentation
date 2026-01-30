python --version
python - << 'EOF'
import torch
print("Torch:", torch.__version__)
print("CUDA :", torch.cuda.is_available())
print("GPU  :", torch.cuda.get_device_name(0))
EOF
