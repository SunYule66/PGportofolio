#!/usr/bin/env python3
"""
Quick script to check GPU availability and PyTorch CUDA support
Usage: python check_gpu.py
"""
import sys

try:
    import torch
    print("=" * 50)
    print("PyTorch GPU Check")
    print("=" * 50)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
        
        # Test tensor creation on GPU
        try:
            x = torch.randn(3, 3).cuda()
            print("\n✓ GPU tensor creation test: PASSED")
        except Exception as e:
            print(f"\n✗ GPU tensor creation test: FAILED - {e}")
    else:
        print("\n⚠ WARNING: CUDA is not available!")
        print("Training will use CPU, which is much slower.")
        print("\nTo use GPU, ensure:")
        print("1. NVIDIA GPU is available")
        print("2. NVIDIA drivers are installed (check: nvidia-smi)")
        print("3. PyTorch with CUDA support is installed")
        print("   Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        sys.exit(1)
    
    print("=" * 50)
    print("✓ GPU check completed successfully!")
    print("=" * 50)
    
except ImportError:
    print("ERROR: PyTorch is not installed!")
    print("Install with: pip install torch")
    sys.exit(1)

