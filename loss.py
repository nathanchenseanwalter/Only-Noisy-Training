import torch
import torch.nn as nn
from loss_utils import mse_loss, stftm_loss, reg_loss

time_loss = mse_loss()
freq_loss = stftm_loss()
reg_loss = reg_loss()

SAMPLE_RATE = 48000
N_FFT = 1022
HOP_LENGTH = 256

def compLossMask(inp, nframes):
    """Compute loss mask efficiently with input validation."""
    if not isinstance(inp, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(inp)}")
        
    if not isinstance(nframes, (list, tuple, torch.Tensor)):
        raise TypeError(f"nframes must be list, tuple or tensor, got {type(nframes)}")
    
    if inp.numel() == 0:
        raise ValueError("Empty input tensor")
        
    # Create mask tensor on same device as input
    loss_mask = torch.zeros_like(inp, requires_grad=False)
    
    # Validate frame lengths
    max_frames = inp.size(-1) if inp.dim() > 0 else 0
    
    for j, seq_len in enumerate(nframes):
        if j >= inp.size(0):
            print(f"Warning: frame index {j} exceeds batch size {inp.size(0)}")
            break
            
        seq_len = int(seq_len)  # Ensure integer
        if seq_len <= 0:
            print(f"Warning: invalid sequence length {seq_len} for sample {j}")
            continue
            
        # Clamp sequence length to valid range
        seq_len = min(seq_len, max_frames)
        loss_mask[j, :, :seq_len] = 1.0
        
    return loss_mask

class RegularizedLoss(nn.Module):
    def __init__(self, gamma=1):
        super().__init__()

        self.gamma = gamma

    '''
    def mseloss(self, image, target):
        x = ((image - target)**2)
        return torch.mean(x)
    '''

    def wsdr_fn(self, x_, y_pred_, y_true_, eps=1e-8):
        """Weighted SDR loss with improved numerical stability and validation."""
        # Input validation
        tensors = [x_, y_pred_, y_true_]
        names = ['x', 'y_pred', 'y_true']
        
        for tensor, name in zip(tensors, names):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be torch.Tensor, got {type(tensor)}")
            if tensor.numel() == 0:
                raise ValueError(f"{name} is empty")
                
        # Flatten tensors
        y_pred = y_pred_.flatten(1)
        y_true = y_true_.flatten(1) 
        x = x_.flatten(1)
        
        # Validate flattened shapes match
        if not (y_pred.shape == y_true.shape == x.shape):
            raise ValueError(f"Shape mismatch after flattening: x={x.shape}, pred={y_pred.shape}, true={y_true.shape}")

        def sdr_fn(true, pred, eps=1e-8):
            """Compute SDR with improved numerical stability."""
            # Clamp to avoid numerical issues
            true = torch.clamp(true, min=-1e6, max=1e6)
            pred = torch.clamp(pred, min=-1e6, max=1e6)
            
            num = torch.sum(true * pred, dim=1)
            
            # More numerically stable norm computation
            true_norm = torch.norm(true, p=2, dim=1)
            pred_norm = torch.norm(pred, p=2, dim=1)
            den = true_norm * pred_norm
            
            # Avoid division by zero
            den = torch.clamp(den, min=eps)
            
            return -(num / den)

        try:
            # Compute noise components
            z_true = x - y_true
            z_pred = x - y_pred

            # Compute weighting factor with numerical stability
            y_true_power = torch.sum(y_true ** 2, dim=1)
            z_true_power = torch.sum(z_true ** 2, dim=1)
            
            total_power = y_true_power + z_true_power + eps
            a = y_true_power / total_power
            
            # Clamp weighting to valid range [0, 1]
            a = torch.clamp(a, min=0.0, max=1.0)

            # Compute SDR components
            sdr_signal = sdr_fn(y_true, y_pred, eps)
            sdr_noise = sdr_fn(z_true, z_pred, eps)
            
            # Weighted combination
            wSDR = a * sdr_signal + (1 - a) * sdr_noise
            
            # Check for valid result
            if not torch.isfinite(wSDR).all():
                print("Warning: Non-finite values in WSDR computation")
                # Return a small penalty instead of invalid values
                return torch.tensor(0.1, device=x.device, requires_grad=True)
                
            return torch.mean(wSDR)
            
        except Exception as e:
            print(f"Error in WSDR computation: {str(e)}")
            return torch.tensor(0.1, device=x.device, requires_grad=True)

    def regloss(self, g1, g2, G1, G2):
        """Compute regularization loss with input validation."""
        tensors = [g1, g2, G1, G2]
        names = ['g1', 'g2', 'G1', 'G2']
        
        # Input validation
        for tensor, name in zip(tensors, names):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be torch.Tensor, got {type(tensor)}")
            if tensor.numel() == 0:
                raise ValueError(f"{name} is empty")
                
        # Check shape compatibility
        shapes = [tensor.shape for tensor in tensors]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError(f"Shape mismatch in regularization loss: {dict(zip(names, shapes))}")
            
        try:
            # Compute regularization term
            diff = g1 - g2 - G1 + G2
            loss = torch.mean(diff ** 2)
            
            # Validate result
            if not torch.isfinite(loss):
                print("Warning: Non-finite regularization loss")
                return torch.tensor(0.0, device=g1.device, requires_grad=True)
                
            return loss
            
        except Exception as e:
            print(f"Error computing regularization loss: {str(e)}")
            return torch.tensor(0.0, device=g1.device, requires_grad=True)

    def forward(self, g1_wav, fg1_wav, g2_wav, g1fx, g2fx):
        """Compute regularized loss with comprehensive input validation."""
        # Input validation
        tensors = [g1_wav, fg1_wav, g2_wav, g1fx, g2fx]
        names = ['g1_wav', 'fg1_wav', 'g2_wav', 'g1fx', 'g2fx']
        
        for tensor, name in zip(tensors, names):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be torch.Tensor, got {type(tensor)}")
            if tensor.numel() == 0:
                raise ValueError(f"{name} is empty")
                
        # Check device consistency
        device = g2_wav.device
        for tensor, name in zip(tensors, names):
            if tensor.device != device:
                raise ValueError(f"Device mismatch: {name} on {tensor.device}, expected {device}")
        
        # Validate shapes are compatible
        batch_size = g2_wav.shape[0]
        if not all(tensor.shape[0] == batch_size for tensor in tensors):
            shapes = [tensor.shape for tensor in tensors]
            raise ValueError(f"Batch size mismatch: {dict(zip(names, shapes))}")
        
        try:
            # Determine frame lengths based on batch size
            if batch_size == 2:
                nframes = [g2_wav.shape[2], g2_wav.shape[2]]
            else:
                nframes = [g2_wav.shape[2]] * batch_size

            # Compute loss mask
            loss_mask = compLossMask(g2_wav, nframes)
            loss_mask = loss_mask.float()
            
            # Check if mask is valid
            if torch.sum(loss_mask) == 0:
                print("Warning: Loss mask is all zeros, using uniform mask")
                loss_mask = torch.ones_like(g2_wav)

            # Compute component losses with error handling
            try:
                loss_time = time_loss(fg1_wav, g2_wav, loss_mask)
            except Exception as e:
                raise RuntimeError(f"Time loss computation failed: {str(e)}")
                
            try:
                loss_freq = freq_loss(fg1_wav, g2_wav, loss_mask)
            except Exception as e:
                raise RuntimeError(f"Frequency loss computation failed: {str(e)}")
                
            # Weighted combination (alpha=0.8, beta=0.2)
            loss1 = (0.8 * loss_time + 0.2 * loss_freq) / 600
            
            # Compute additional loss components
            try:
                wsdr_loss = self.wsdr_fn(g1_wav, fg1_wav, g2_wav)
            except Exception as e:
                print(f"Warning: WSDR loss computation failed: {str(e)}, using zero")
                wsdr_loss = torch.tensor(0.0, device=device, requires_grad=True)
                
            try:
                reg_loss_val = self.regloss(fg1_wav, g2_wav, g1fx, g2fx)
            except Exception as e:
                print(f"Warning: Regularization loss computation failed: {str(e)}, using zero")
                reg_loss_val = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Final loss combination
            total_loss = loss1 + wsdr_loss + self.gamma * reg_loss_val
            
            # Validate final loss
            if not torch.isfinite(total_loss):
                print(f"Warning: Non-finite loss detected: {total_loss.item()}")
                # Return a small positive loss to prevent training breakdown
                return torch.tensor(1e-6, device=device, requires_grad=True)
                
            return total_loss
            
        except Exception as e:
            print(f"Error in loss computation: {str(e)}")
            # Return fallback loss
            return torch.tensor(1e-6, device=device, requires_grad=True)
