#!/bin/bash
# GPU Training Script for AutoDL
# Usage: ./train_gpu.sh

echo "=========================================="
echo "PGPortfolio GPU Training Script"
echo "=========================================="

# Check if GPU is available
echo ""
echo "Checking GPU availability..."
python3 -c "import torch; \
    cuda_available = torch.cuda.is_available(); \
    print(f'CUDA Available: {cuda_available}'); \
    if cuda_available: \
        print(f'GPU Device: {torch.cuda.get_device_name(0)}'); \
        print(f'CUDA Version: {torch.version.cuda}'); \
        print(f'GPU Count: {torch.cuda.device_count()}'); \
    else: \
        print('WARNING: CUDA is not available! Training will use CPU.'); \
        exit(1)"

if [ $? -ne 0 ]; then
    echo "ERROR: GPU check failed. Please ensure PyTorch with CUDA support is installed."
    exit 1
fi

# Check if database exists
if [ ! -f "database/okx_data.db" ]; then
    echo ""
    echo "WARNING: database/okx_data.db not found!"
    echo "Please ensure your database file is in the database/ directory."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if config exists
if [ ! -f "pgportfolio/net_config.json" ]; then
    echo "ERROR: pgportfolio/net_config.json not found!"
    exit 1
fi

echo ""
echo "Starting training with GPU..."
echo "Training logs will be saved to: train_package/*/programlog"
echo "Press Ctrl+C to stop training"
echo ""

# Start training
python main.py --mode train --device gpu

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="

