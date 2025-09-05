"""
Basic tests for dataset utility functions.
Tests core functionality before refactoring.
"""
import torch
import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from dataset_utils import subsample2, subsample4, SpeechDataset


class TestSubsampleFunctions:
    """Test subsampling functions"""
    
    def test_subsample2_shape(self):
        """Test subsample2 maintains correct output shape"""
        # Create test tensor [channels, samples]
        channels, samples = 1, 66000  # Enough samples for subsampling
        wav = torch.randn(channels, samples)
        
        wav1, wav2 = subsample2(wav)
        
        expected_dim = samples // 2 - 128
        assert wav1.shape == (channels, expected_dim)
        assert wav2.shape == (channels, expected_dim)
    
    def test_subsample4_shape(self):
        """Test subsample4 maintains correct output shape"""
        channels, samples = 1, 66000
        wav = torch.randn(channels, samples)
        
        wav1, wav2 = subsample4(wav)
        
        expected_dim = samples // 4 - 192
        assert wav1.shape == (channels, expected_dim)
        assert wav2.shape == (channels, expected_dim)
    
    def test_subsample2_device_preservation(self):
        """Test that subsample2 preserves device placement"""
        if torch.cuda.is_available():
            wav = torch.randn(1, 66000).cuda()
            wav1, wav2 = subsample2(wav)
            assert wav1.device == wav.device
            assert wav2.device == wav.device
    
    def test_subsample2_deterministic_seeding(self):
        """Test that subsample2 produces consistent results with same seed"""
        wav = torch.randn(1, 66000)
        
        # Set seed and run
        np.random.seed(42)
        wav1_a, wav2_a = subsample2(wav)
        
        # Reset seed and run again
        np.random.seed(42)
        wav1_b, wav2_b = subsample2(wav)
        
        # Results should be identical
        torch.testing.assert_close(wav1_a, wav1_b)
        torch.testing.assert_close(wav2_a, wav2_b)


class TestSpeechDataset:
    """Test SpeechDataset class (without requiring actual files)"""
    
    def test_dataset_init(self):
        """Test dataset initialization"""
        noisy_files = ["test1.wav", "test2.wav"]
        clean_files = ["clean1.wav", "clean2.wav"]
        
        dataset = SpeechDataset(noisy_files, clean_files)
        
        assert len(dataset) == 2
        assert dataset.n_fft == 1022
        assert dataset.hop_length == 256
        assert dataset.max_len == 65280
    
    def test_prepare_sample_padding(self):
        """Test sample preparation and padding"""
        dataset = SpeechDataset([], [])  # Empty lists for testing
        
        # Test with short waveform
        short_wave = torch.randn(1, 1000)
        padded = dataset._prepare_sample(short_wave)
        
        assert padded.shape == (1, 65280)
        # Check that original data is at the end
        assert torch.allclose(padded[0, -1000:], short_wave[0, :])
    
    def test_prepare_sample_truncation(self):
        """Test sample preparation with truncation"""
        dataset = SpeechDataset([], [])
        
        # Test with long waveform (longer than max_len)
        long_wave = torch.randn(1, 100000)
        truncated = dataset._prepare_sample(long_wave)
        
        assert truncated.shape == (1, 65280)
        # Check truncation happened correctly
        assert torch.allclose(truncated[0, :], long_wave[0, :65280])


def test_constants():
    """Test that constants are properly defined"""
    from dataset_utils import SAMPLE_RATE, N_FFT, HOP_LENGTH
    
    assert SAMPLE_RATE == 48000
    assert N_FFT == 1022
    assert HOP_LENGTH == 256


if __name__ == "__main__":
    # Run basic smoke tests
    print("Running basic dataset utility tests...")
    
    # Test subsample functions
    test_tensor = torch.randn(1, 66000)
    wav1, wav2 = subsample2(test_tensor)
    print(f"subsample2: input {test_tensor.shape} -> output {wav1.shape}, {wav2.shape}")
    
    wav1, wav2 = subsample4(test_tensor)
    print(f"subsample4: input {test_tensor.shape} -> output {wav1.shape}, {wav2.shape}")
    
    # Test dataset class
    dataset = SpeechDataset([], [])
    test_wave = torch.randn(1, 5000)
    prepared = dataset._prepare_sample(test_wave)
    print(f"Sample preparation: {test_wave.shape} -> {prepared.shape}")
    
    print("All basic tests passed!")