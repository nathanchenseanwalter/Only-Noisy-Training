import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader

SAMPLE_RATE = 48000
N_FFT = 1022
HOP_LENGTH = 256

def subsample4(wav):  
    """Optimized vectorized subsampling for k=4 with error handling."""
    if not isinstance(wav, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(wav)}")
    
    if wav.dim() != 2:
        raise ValueError(f"Expected 2D tensor (channels, samples), got {wav.dim()}D")
        
    k = 4
    channels, dim = wav.shape
    
    if dim < k:
        raise ValueError(f"Input length {dim} is too short for subsampling factor {k}")
    
    # Ensure dim is divisible by k for clean indexing
    usable_dim = (dim // k) * k
    if usable_dim < 192 * k:
        raise ValueError(f"Insufficient samples for subsampling: need at least {192 * k}, got {usable_dim}")
    
    dim1 = usable_dim // k - 192
    device = wav.device
    
    # Vectorized approach - reshape and select randomly
    wav_reshaped = wav[:, :usable_dim].view(channels, -1, k)  # [channels, dim1+192, k]
    wav_working = wav_reshaped[:, :dim1+192, :]  # [channels, dim1+192, k]
    
    # Generate random indices for each position
    random_indices = torch.randint(0, k, (dim1,), device=device)
    
    # Vectorized selection using advanced indexing
    arange_dim1 = torch.arange(dim1, device=device)
    wav1 = wav_working[:, arange_dim1, random_indices]  # [channels, dim1]
    
    # For wav2, select the next index (with wraparound)
    next_indices = (random_indices + 1) % k
    wav2 = wav_working[:, arange_dim1, next_indices]  # [channels, dim1]
    
    return wav1, wav2

def subsample2(wav):  
    """Optimized vectorized subsampling for k=2 with error handling."""
    if not isinstance(wav, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(wav)}")
    
    if wav.dim() != 2:
        raise ValueError(f"Expected 2D tensor (channels, samples), got {wav.dim()}D")
        
    k = 2
    channels, dim = wav.shape
    
    if dim < k:
        raise ValueError(f"Input length {dim} is too short for subsampling factor {k}")
    
    # Ensure dim is divisible by k for clean indexing
    usable_dim = (dim // k) * k
    if usable_dim < 128 * k:
        raise ValueError(f"Insufficient samples for subsampling: need at least {128 * k}, got {usable_dim}")
    
    dim1 = usable_dim // k - 128
    device = wav.device
    
    # Vectorized approach - reshape and select randomly
    wav_reshaped = wav[:, :usable_dim].view(channels, -1, k)  # [channels, dim1+128, k]
    wav_working = wav_reshaped[:, :dim1+128, :]  # [channels, dim1+128, k]
    
    # Generate random binary choices for each position
    random_choices = torch.randint(0, 2, (dim1,), device=device, dtype=torch.bool)
    
    # Vectorized selection
    arange_dim1 = torch.arange(dim1, device=device)
    wav1 = torch.where(random_choices, 
                      wav_working[:, arange_dim1, 0],   # select first element
                      wav_working[:, arange_dim1, 1])   # select second element
    wav2 = torch.where(random_choices,
                      wav_working[:, arange_dim1, 1],   # select second element  
                      wav_working[:, arange_dim1, 0])   # select first element
    
    return wav1, wav2

class SpeechDataset(Dataset):
    def __init__(self, noisy_files, clean_files, n_fft=N_FFT, hop_length=HOP_LENGTH):
        super().__init__()
        # list of files
        self.noisy_files = sorted(noisy_files)
        self.clean_files = sorted(clean_files)

        # stft parameters
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.len_ = len(self.noisy_files)

        # fixed len
        self.max_len = 65280

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        waveform, _ = torchaudio.load(file)
        return waveform

    def _prepare_sample(self, waveform):
        """Optimized tensor-native sample preparation with error handling."""
        if not isinstance(waveform, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(waveform)}")
            
        if waveform.dim() != 2 or waveform.size(0) != 1:
            raise ValueError(f"Expected waveform shape [1, samples], got {waveform.shape}")
            
        current_len = waveform.size(1)
        
        # Direct tensor operations without numpy conversion
        if current_len >= self.max_len:
            # Truncate if too long
            output = waveform[:, :self.max_len]
        else:
            # Pad if too short - use zeros and fill from the right
            output = torch.zeros((1, self.max_len), dtype=waveform.dtype, device=waveform.device)
            output[0, -current_len:] = waveform[0, :]
        
        return output

    def __getitem__(self, index):
        """Optimized data loading with error handling and memory efficiency."""
        try:
            # Input validation
            if not (0 <= index < len(self.noisy_files)):
                raise IndexError(f"Index {index} out of range [0, {len(self.noisy_files)})")
                
            # load to tensors and normalization
            x_clean = self.load_sample(self.clean_files[index])
            x_noisy = self.load_sample(self.noisy_files[index])
            
            # Validate loaded samples
            if x_clean.numel() == 0 or x_noisy.numel() == 0:
                raise ValueError(f"Empty audio file at index {index}")

            # padding/cutting
            x_clean = self._prepare_sample(x_clean)
            x_noisy = self._prepare_sample(x_noisy)

            # compute inputs and targets (g1x and g2x are subsampled form x_noisy)
            g1_wav, g2_wav = subsample2(x_noisy)
            g1_wav, g2_wav = g1_wav.float(), g2_wav.float()
            
            # Optimized STFT computation - reuse window and reduce memory
            device = x_noisy.device
            window = torch.hann_window(self.n_fft, device=device)
            
            def compute_stft_realimag(signal):
                """Helper to compute STFT and convert to real/imag format efficiently."""
                if signal.numel() == 0:
                    raise ValueError("Cannot compute STFT on empty signal")
                    
                stft_complex = torch.stft(
                    input=signal, 
                    n_fft=self.n_fft, 
                    hop_length=self.hop_length, 
                    normalized=True, 
                    return_complex=True, 
                    window=window
                )
                # Direct stack operation is more memory efficient
                return torch.stack([stft_complex.real, stft_complex.imag], dim=-1)
            
            # Compute STFTs with error handling
            x_noisy_stft = compute_stft_realimag(x_noisy)
            x_clean_stft = compute_stft_realimag(x_clean) 
            g1_stft = compute_stft_realimag(g1_wav)
            
            # Validate output shapes
            expected_stft_shape = (1, self.n_fft//2 + 1, -1, 2)  # -1 for time dimension
            if not all(stft.shape[:2] == expected_stft_shape[:2] and stft.shape[3] == 2 
                      for stft in [x_noisy_stft, x_clean_stft, g1_stft]):
                raise RuntimeError(f"STFT computation resulted in unexpected shapes")
                
            return x_noisy_stft, g1_stft, g1_wav, g2_wav, x_clean_stft
            
        except Exception as e:
            raise RuntimeError(f"Error processing sample at index {index}: {str(e)}") from e
