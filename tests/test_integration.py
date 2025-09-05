"""
Integration tests to validate core functionality after refactoring.
Tests model loading, forward pass, and training components.
"""
import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from DCUnet10_TSTM.DCUnet import DCUnet10, DCUnet10_rTSTM, DCUnet10_cTSTM
from dataset_utils import subsample2, SpeechDataset
from loss import RegularizedLoss


def test_model_instantiation():
    """Test that all model variants can be instantiated"""
    print("Testing model instantiation...")
    
    N_FFT = 1022
    HOP_LENGTH = 256
    
    # Test basic model
    model1 = DCUnet10(N_FFT, HOP_LENGTH)
    print(f"[OK] DCUnet10 instantiated")
    
    # Test TSTM variants
    model2 = DCUnet10_rTSTM(N_FFT, HOP_LENGTH)
    print(f"[OK] DCUnet10_rTSTM instantiated")
    
    model3 = DCUnet10_cTSTM(N_FFT, HOP_LENGTH)
    print(f"[OK] DCUnet10_cTSTM instantiated")
    
    return model1


def test_forward_pass():
    """Test forward pass with dummy data"""
    print("Testing forward pass...")
    
    N_FFT = 1022
    HOP_LENGTH = 256
    
    model = DCUnet10(N_FFT, HOP_LENGTH)
    
    # Create dummy STFT input [batch, channels, freq, time, complex]
    batch_size = 1
    channels = 1
    freq_bins = N_FFT // 2 + 1
    time_frames = 256
    
    x = torch.randn(batch_size, channels, freq_bins, time_frames, 2)
    
    # Test forward pass
    output = model(x, n_fft=N_FFT, hop_length=HOP_LENGTH, is_istft=True)
    print(f"[OK] Forward pass successful, output shape: {output.shape}")
    
    return output


def test_loss_computation():
    """Test loss function computation"""
    print("Testing loss computation...")
    
    loss_fn = RegularizedLoss(gamma=1.0)
    
    # Create dummy audio tensors
    batch_size, channels, samples = 2, 1, 32512
    g1_wav = torch.randn(batch_size, channels, samples)
    fg1_wav = torch.randn(batch_size, channels, samples)
    g2_wav = torch.randn(batch_size, channels, samples)
    g1fx = torch.randn(batch_size, channels, samples)
    g2fx = torch.randn(batch_size, channels, samples)
    
    loss = loss_fn(g1_wav, fg1_wav, g2_wav, g1fx, g2fx)
    print(f"[OK] Loss computation successful, value: {loss.item():.6f}")
    
    return loss


def test_subsampling():
    """Test subsampling functions"""
    print("Testing subsampling...")
    
    # Create test audio tensor
    wav = torch.randn(1, 66000)
    
    # Test subsample2
    wav1, wav2 = subsample2(wav)
    print(f"[OK] subsample2: {wav.shape} -> {wav1.shape}, {wav2.shape}")
    
    # Verify output shapes are correct
    expected_dim = 66000 // 2 - 128
    assert wav1.shape == (1, expected_dim), f"Expected {(1, expected_dim)}, got {wav1.shape}"
    assert wav2.shape == (1, expected_dim), f"Expected {(1, expected_dim)}, got {wav2.shape}"
    
    return wav1, wav2


def test_device_compatibility():
    """Test device compatibility (CPU/CUDA)"""
    print("Testing device compatibility...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    N_FFT = 1022
    HOP_LENGTH = 256
    
    model = DCUnet10(N_FFT, HOP_LENGTH).to(device)
    loss_fn = RegularizedLoss().to(device)
    
    # Create dummy data on same device
    batch_size = 1
    channels = 1
    freq_bins = N_FFT // 2 + 1
    time_frames = 128  # Smaller for testing
    
    x = torch.randn(batch_size, channels, freq_bins, time_frames, 2, device=device)
    
    # Test forward pass
    with torch.no_grad():
        output = model(x, n_fft=N_FFT, hop_length=HOP_LENGTH, is_istft=True)
    
    print(f"[OK] Device compatibility test passed on {device}")
    return output


def run_integration_tests():
    """Run all integration tests"""
    print("=" * 60)
    print("RUNNING INTEGRATION TESTS AFTER REFACTORING")
    print("=" * 60)
    
    try:
        # Test 1: Model instantiation
        model = test_model_instantiation()
        print()
        
        # Test 2: Forward pass
        output = test_forward_pass()
        print()
        
        # Test 3: Loss computation
        loss = test_loss_computation()
        print()
        
        # Test 4: Subsampling
        wav1, wav2 = test_subsampling()
        print()
        
        # Test 5: Device compatibility
        device_output = test_device_compatibility()
        print()
        
        print("=" * 60)
        print("[SUCCESS] ALL INTEGRATION TESTS PASSED!")
        print("[SUCCESS] Refactoring preserved core functionality")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print("=" * 60)
        print(f"[FAIL] INTEGRATION TEST FAILED: {str(e)}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    run_integration_tests()