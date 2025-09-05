import torch
from preprocess import TorchSignalToFrames
from scipy import linalg
import numpy as np
import scipy

class mse_loss(object):
    """Memory-optimized MSE loss with input validation."""
    
    def __call__(self, outputs, labels, loss_mask):
        # Input validation
        if not all(isinstance(x, torch.Tensor) for x in [outputs, labels, loss_mask]):
            raise TypeError("All inputs must be torch.Tensor")
            
        if not (outputs.shape == labels.shape == loss_mask.shape):
            raise ValueError(f"Shape mismatch: outputs {outputs.shape}, labels {labels.shape}, mask {loss_mask.shape}")
        
        # Check for valid mask
        mask_sum = torch.sum(loss_mask)
        if mask_sum == 0:
            print("Warning: Loss mask is all zeros, returning zero loss")
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)
            
        # Memory-efficient computation (avoid intermediate tensor storage)
        diff = outputs - labels
        masked_diff_squared = (diff * loss_mask) ** 2.0
        loss = torch.sum(masked_diff_squared) / mask_sum
        
        return loss


class stftm_loss(object):
    """Memory-optimized STFT magnitude loss with error handling."""
    
    def __init__(self, frame_size=512, frame_shift=256, loss_type='mae'):
        if loss_type not in ['mse', 'mae']:
            raise ValueError(f"loss_type must be 'mse' or 'mae', got {loss_type}")
            
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.loss_type = loss_type
        
        try:
            self.frame = TorchSignalToFrames(frame_size=self.frame_size,
                                           frame_shift=self.frame_shift)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize framing: {str(e)}")
            
        # Pre-compute DFT matrices
        try:
            D = linalg.dft(frame_size)
            W = np.hamming(self.frame_size)
            DR = np.real(D)
            DI = np.imag(D)
            
            self.DR = torch.from_numpy(DR).float().contiguous().transpose(0, 1)
            self.DI = torch.from_numpy(DI).float().contiguous().transpose(0, 1)
            self.W = torch.from_numpy(W).float()
        except Exception as e:
            raise RuntimeError(f"Failed to compute DFT matrices: {str(e)}")
            
        self._device_initialized = False
        self._current_device = None

    def _ensure_device_compatibility(self, device):
        """Ensure all tensors are on the correct device."""
        if not self._device_initialized or self._current_device != device:
            try:
                self.DR = self.DR.to(device)
                self.DI = self.DI.to(device) 
                self.W = self.W.to(device)
                self._device_initialized = True
                self._current_device = device
            except Exception as e:
                raise RuntimeError(f"Failed to move tensors to device {device}: {str(e)}")

    def __call__(self, outputs, labels, loss_mask):
        # Input validation
        if not all(isinstance(x, torch.Tensor) for x in [outputs, labels, loss_mask]):
            raise TypeError("All inputs must be torch.Tensor")
            
        if outputs.numel() == 0 or labels.numel() == 0:
            raise ValueError("Empty input tensors")
            
        device = outputs.device
        self._ensure_device_compatibility(device)
        
        try:
            # Frame the signals
            outputs_framed = self.frame(outputs)
            labels_framed = self.frame(labels)
            loss_mask_framed = self.frame(loss_mask)
            
            # Compute STFT magnitudes
            outputs_stftm = self.get_stftm(outputs_framed)
            labels_stftm = self.get_stftm(labels_framed)
            
            # Check for valid mask
            mask_sum = torch.sum(loss_mask_framed)
            if mask_sum == 0:
                print("Warning: STFT loss mask is all zeros, returning zero loss")
                return torch.tensor(0.0, device=device, requires_grad=True)
            
            # Memory-efficient masked computation
            if self.loss_type == 'mse':
                diff = outputs_stftm - labels_stftm
                masked_diff_squared = (diff * loss_mask_framed) ** 2
                loss = torch.sum(masked_diff_squared) / mask_sum
            elif self.loss_type == 'mae':
                diff = torch.abs(outputs_stftm - labels_stftm)
                masked_diff = diff * loss_mask_framed
                loss = torch.sum(masked_diff) / mask_sum

            return loss
            
        except Exception as e:
            raise RuntimeError(f"Error computing STFT loss: {str(e)}")

    def get_stftm(self, frames):
        """Compute STFT magnitude efficiently."""
        try:
            # Apply window
            windowed_frames = frames * self.W
            
            # Compute STFT (real and imaginary parts)
            stft_R = torch.matmul(windowed_frames, self.DR)
            stft_I = torch.matmul(windowed_frames, self.DI)
            
            # Compute magnitude more efficiently
            stftm = torch.sqrt(stft_R**2 + stft_I**2 + 1e-8)  # Add epsilon for numerical stability
            
            return stftm
            
        except Exception as e:
            raise RuntimeError(f"Error computing STFT magnitude: {str(e)}")

    def get_stftm(self, frames):
        frames = frames * self.W
        stft_R = torch.matmul(frames, self.DR)
        stft_I = torch.matmul(frames, self.DI)
        stftm = torch.abs(stft_R) + torch.abs(stft_I)
        return stftm

class reg_loss(object):
    """Memory-optimized regularization loss with input validation."""
    
    def __call__(self, fg1, g2, g1fx, g2fx):
        # Input validation
        if not all(isinstance(x, torch.Tensor) for x in [fg1, g2, g1fx, g2fx]):
            raise TypeError("All inputs must be torch.Tensor")
            
        # Check shapes are compatible
        shapes = [x.shape for x in [fg1, g2, g1fx, g2fx]]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError(f"Shape mismatch in regularization loss: {shapes}")
            
        if fg1.numel() == 0:
            print("Warning: Empty tensors in regularization loss, returning zero")
            return torch.tensor(0.0, device=fg1.device, requires_grad=True)
        
        # Memory-efficient computation
        diff = fg1 - g2 - g1fx + g2fx
        return torch.mean(diff**2)