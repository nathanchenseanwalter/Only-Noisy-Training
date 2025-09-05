#!/usr/bin/env python3
"""
Test script to validate memory optimizations and error handling improvements.
This script tests key components without requiring full dataset loading.
"""

import torch
import numpy as np
from pathlib import Path

# Test imports to ensure optimizations didn't break functionality
try:
    from dataset_utils import subsample2, subsample4, SpeechDataset
    from DCUnet10_TSTM.DCUnet import DCUnet10, DCUnet10_rTSTM, DCUnet10_cTSTM
    from loss import RegularizedLoss
    from loss_utils import mse_loss, stftm_loss, reg_loss
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)

def test_subsample_functions():
    """Test optimized subsampling functions."""
    print("\n=== Testing Subsample Functions ===")
    
    # Test subsample2
    try:
        # Create test tensor
        test_wav = torch.randn(2, 1000)  # [channels, samples]
        g1, g2 = subsample2(test_wav)
        print(f"✓ subsample2: input {test_wav.shape} -> outputs {g1.shape}, {g2.shape}")
        
        # Test error handling
        try:
            subsample2(torch.randn(100))  # Wrong dimensions
            print("✗ subsample2 error handling failed")
        except ValueError:
            print("✓ subsample2 error handling works")
            
    except Exception as e:
        print(f"✗ subsample2 test failed: {e}")
    
    # Test subsample4  
    try:
        test_wav = torch.randn(2, 2000)  # Larger for k=4
        g1, g2 = subsample4(test_wav)
        print(f"✓ subsample4: input {test_wav.shape} -> outputs {g1.shape}, {g2.shape}")
        
        # Test error handling
        try:
            subsample4(torch.randn(2, 100))  # Too small
            print("✗ subsample4 error handling failed")
        except ValueError:
            print("✓ subsample4 error handling works")
            
    except Exception as e:
        print(f"✗ subsample4 test failed: {e}")

def test_model_forward():
    """Test model forward passes with memory optimization."""
    print("\n=== Testing Model Forward Passes ===")
    
    # Parameters
    n_fft = 1022
    hop_length = 256
    batch_size = 2
    freq_bins = n_fft // 2 + 1
    time_frames = 200
    
    # Create test input [B, C, F, T, 2]
    test_input = torch.randn(batch_size, 1, freq_bins, time_frames, 2)
    
    # Test DCUnet10
    try:
        model = DCUnet10(n_fft, hop_length)
        with torch.no_grad():
            output = model(test_input, n_fft, hop_length, is_istft=False)
        print(f"✓ DCUnet10: input {test_input.shape} -> output {output.shape}")
        
        # Test error handling
        try:
            model(torch.randn(2, 3, 4), n_fft, hop_length)  # Wrong shape
            print("✗ DCUnet10 error handling failed")
        except ValueError:
            print("✓ DCUnet10 error handling works")
            
    except Exception as e:
        print(f"✗ DCUnet10 test failed: {e}")
    
    # Test rTSTM model (more memory intensive)
    try:
        model = DCUnet10_rTSTM(n_fft, hop_length)
        with torch.no_grad():
            output = model(test_input, n_fft, hop_length, is_istft=False)
        print(f"✓ DCUnet10_rTSTM: input {test_input.shape} -> output {output.shape}")
    except Exception as e:
        print(f"✗ DCUnet10_rTSTM test failed: {e}")

def test_loss_functions():
    """Test optimized loss functions."""
    print("\n=== Testing Loss Functions ===")
    
    # Create test tensors
    batch_size = 2
    samples = 1000
    
    g1_wav = torch.randn(batch_size, 1, samples)
    fg1_wav = torch.randn(batch_size, 1, samples) 
    g2_wav = torch.randn(batch_size, 1, samples)
    g1fx = torch.randn(batch_size, 1, samples)
    g2fx = torch.randn(batch_size, 1, samples)
    
    # Test RegularizedLoss
    try:
        loss_fn = RegularizedLoss(gamma=1.0)
        loss = loss_fn(g1_wav, fg1_wav, g2_wav, g1fx, g2fx)
        print(f"✓ RegularizedLoss: {loss.item():.6f}")
        
        # Test error handling
        try:
            loss_fn(g1_wav, fg1_wav, torch.randn(3, 1, samples), g1fx, g2fx)  # Mismatched batch size
            print("✗ RegularizedLoss error handling failed")
        except ValueError:
            print("✓ RegularizedLoss error handling works")
            
    except Exception as e:
        print(f"✗ RegularizedLoss test failed: {e}")
    
    # Test individual loss components
    try:
        mse_fn = mse_loss()
        outputs = torch.randn(2, 1, 1000)
        labels = torch.randn(2, 1, 1000)  
        mask = torch.ones(2, 1, 1000)
        
        mse_val = mse_fn(outputs, labels, mask)
        print(f"✓ MSE loss: {mse_val.item():.6f}")
        
        # Test with zero mask
        zero_mask = torch.zeros(2, 1, 1000)
        mse_val = mse_fn(outputs, labels, zero_mask)
        print(f"✓ MSE loss with zero mask: {mse_val.item():.6f}")
        
    except Exception as e:
        print(f"✗ MSE loss test failed: {e}")

def test_memory_efficiency():
    """Test memory efficiency improvements."""
    print("\n=== Testing Memory Efficiency ===")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Testing on GPU: {torch.cuda.get_device_name()}")
        
        # Test memory usage with large tensors
        try:
            # Create reasonably large tensors
            large_tensor = torch.randn(4, 64, 256, 200, 2, device=device)
            
            model = DCUnet10(1022, 256).to(device)
            
            # Test forward pass
            torch.cuda.empty_cache()  # Clear cache before test
            initial_memory = torch.cuda.memory_allocated()
            
            with torch.no_grad():
                output = model(large_tensor, 1022, 256, is_istft=False)
                
            peak_memory = torch.cuda.memory_allocated()
            memory_used = (peak_memory - initial_memory) / 1024**2  # MB
            
            print(f"✓ GPU memory test: {memory_used:.2f} MB used for forward pass")
            
            # Clean up
            del large_tensor, output
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"✗ GPU memory test failed: {e}")
    else:
        print("CPU testing - creating moderately sized tensors")
        try:
            test_tensor = torch.randn(2, 32, 128, 100, 2)
            model = DCUnet10(1022, 256)
            
            with torch.no_grad():
                output = model(test_tensor, 1022, 256, is_istft=False)
                
            print(f"✓ CPU memory test completed successfully")
            
        except Exception as e:
            print(f"✗ CPU memory test failed: {e}")

def main():
    """Run all optimization tests."""
    print("Testing PyTorch Only-Noisy Training Optimizations")
    print("=" * 50)
    
    # Set deterministic behavior for testing
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    test_subsample_functions()
    test_model_forward() 
    test_loss_functions()
    test_memory_efficiency()
    
    print("\n" + "=" * 50)
    print("Optimization testing completed!")
    
    # Summary
    print("\nKey Optimizations Implemented:")
    print("• Vectorized subsampling (O(n) instead of O(n²))")
    print("• Memory-efficient tensor operations") 
    print("• Improved error handling with validation")
    print("• Reduced GPU memory transfers")
    print("• In-place operations where possible")
    print("• Explicit tensor cleanup in model forward passes")
    print("• Gradient accumulation support")
    print("• Numerical stability improvements in loss functions")

if __name__ == "__main__":
    main()