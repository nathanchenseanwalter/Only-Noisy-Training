"""
Basic tests for loss functions.
Tests core functionality before refactoring.
"""
import torch
import torch.nn as nn
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from loss import RegularizedLoss, compLossMask


class TestLossFunctions:
    """Test loss function implementations"""
    
    def test_regularized_loss_init(self):
        """Test RegularizedLoss initialization"""
        loss_fn = RegularizedLoss(gamma=2.0)
        assert loss_fn.gamma == 2.0
        
        # Test default gamma
        loss_fn_default = RegularizedLoss()
        assert loss_fn_default.gamma == 1.0
    
    def test_wsdr_function_shapes(self):
        """Test wSDR function with correct tensor shapes"""
        loss_fn = RegularizedLoss()
        
        batch_size, channels, samples = 2, 1, 32512
        g1_wav = torch.randn(batch_size, channels, samples)
        fg1_wav = torch.randn(batch_size, channels, samples)
        g2_wav = torch.randn(batch_size, channels, samples)
        
        # Should not raise an error
        wsdr_loss = loss_fn.wsdr_fn(g1_wav, fg1_wav, g2_wav)
        assert wsdr_loss.shape == torch.Size([])  # Scalar loss
    
    def test_regloss_function(self):
        """Test regularization loss function"""
        loss_fn = RegularizedLoss()
        
        batch_size, channels, samples = 2, 1, 32512
        g1 = torch.randn(batch_size, channels, samples)
        g2 = torch.randn(batch_size, channels, samples)
        G1 = torch.randn(batch_size, channels, samples)
        G2 = torch.randn(batch_size, channels, samples)
        
        reg_loss = loss_fn.regloss(g1, g2, G1, G2)
        assert reg_loss.shape == torch.Size([])  # Scalar loss
        assert reg_loss >= 0  # MSE should be non-negative
    
    def test_loss_forward_pass(self):
        """Test complete forward pass of RegularizedLoss"""
        loss_fn = RegularizedLoss(gamma=1.0)
        
        batch_size, channels, samples = 2, 1, 32512
        g1_wav = torch.randn(batch_size, channels, samples)
        fg1_wav = torch.randn(batch_size, channels, samples)
        g2_wav = torch.randn(batch_size, channels, samples)
        g1fx = torch.randn(batch_size, channels, samples)
        g2fx = torch.randn(batch_size, channels, samples)
        
        # Forward pass should not raise errors
        total_loss = loss_fn(g1_wav, fg1_wav, g2_wav, g1fx, g2fx)
        assert total_loss.shape == torch.Size([])  # Scalar loss
        assert torch.isfinite(total_loss)  # Should be finite


class TestLossMask:
    """Test loss mask computation"""
    
    def test_comp_loss_mask_batch2(self):
        """Test loss mask computation for batch size 2"""
        batch_size, channels, samples = 2, 1, 32512
        inp = torch.zeros(batch_size, channels, samples)
        nframes = [32512, 32512]
        
        loss_mask = compLossMask(inp, nframes)
        
        assert loss_mask.shape == inp.shape
        assert loss_mask.requires_grad == False
        # Check that mask is properly set
        assert torch.all(loss_mask[:, :, :32512] == 1.0)
    
    def test_comp_loss_mask_batch1(self):
        """Test loss mask computation for batch size 1"""
        batch_size, channels, samples = 1, 1, 32512
        inp = torch.zeros(batch_size, channels, samples)
        nframes = [32512]
        
        loss_mask = compLossMask(inp, nframes)
        
        assert loss_mask.shape == inp.shape
        assert loss_mask.requires_grad == False
        assert torch.all(loss_mask[0, :, :32512] == 1.0)


def test_loss_device_compatibility():
    """Test that loss functions work with CUDA if available"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    loss_fn = RegularizedLoss().to(device)
    
    batch_size, channels, samples = 1, 1, 16000  # Smaller for testing
    g1_wav = torch.randn(batch_size, channels, samples, device=device)
    fg1_wav = torch.randn(batch_size, channels, samples, device=device)
    g2_wav = torch.randn(batch_size, channels, samples, device=device)
    g1fx = torch.randn(batch_size, channels, samples, device=device)
    g2fx = torch.randn(batch_size, channels, samples, device=device)
    
    loss = loss_fn(g1_wav, fg1_wav, g2_wav, g1fx, g2fx)
    assert loss.device == device


if __name__ == "__main__":
    # Run basic smoke tests
    print("Running basic loss function tests...")
    
    # Test loss initialization
    loss_fn = RegularizedLoss(gamma=2.0)
    print(f"RegularizedLoss initialized with gamma={loss_fn.gamma}")
    
    # Test forward pass with dummy data
    batch_size, channels, samples = 2, 1, 32512
    g1_wav = torch.randn(batch_size, channels, samples)
    fg1_wav = torch.randn(batch_size, channels, samples)
    g2_wav = torch.randn(batch_size, channels, samples)
    g1fx = torch.randn(batch_size, channels, samples)
    g2fx = torch.randn(batch_size, channels, samples)
    
    total_loss = loss_fn(g1_wav, fg1_wav, g2_wav, g1fx, g2fx)
    print(f"Forward pass completed, loss value: {total_loss.item():.6f}")
    
    # Test loss mask
    inp = torch.zeros(2, 1, 32512)
    nframes = [32512, 32512]
    mask = compLossMask(inp, nframes)
    print(f"Loss mask shape: {mask.shape}")
    
    print("All basic tests passed!")