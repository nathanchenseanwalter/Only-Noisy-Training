import torch
import torch.nn as nn
from DCUnet10_TSTM.Dual_Transformer import Dual_Transformer

class CConv2d(nn.Module):
    """
    Class of complex valued convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        self.real_conv = nn.Conv2d(in_channels=self.in_channels, 
                                   out_channels=self.out_channels, 
                                   kernel_size=self.kernel_size, 
                                   padding=self.padding, 
                                   stride=self.stride)
        
        self.im_conv = nn.Conv2d(in_channels=self.in_channels, 
                                 out_channels=self.out_channels, 
                                 kernel_size=self.kernel_size, 
                                 padding=self.padding, 
                                 stride=self.stride)
        
        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)
        
        
    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]
        
        c_real = self.real_conv(x_real) - self.im_conv(x_im)
        c_im = self.im_conv(x_real) + self.real_conv(x_im)
        
        output = torch.stack([c_real, c_im], dim=-1)
        return output

class CConvTranspose2d(nn.Module):
    """
      Class of complex valued dilation convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0, padding=0):
        super().__init__()
        
        self.in_channels = in_channels

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_padding = output_padding
        self.padding = padding
        self.stride = stride
        
        self.real_convt = nn.ConvTranspose2d(in_channels=self.in_channels, 
                                            out_channels=self.out_channels, 
                                            kernel_size=self.kernel_size, 
                                            output_padding=self.output_padding,
                                            padding=self.padding,
                                            stride=self.stride)
        
        self.im_convt = nn.ConvTranspose2d(in_channels=self.in_channels, 
                                            out_channels=self.out_channels, 
                                            kernel_size=self.kernel_size, 
                                            output_padding=self.output_padding, 
                                            padding=self.padding,
                                            stride=self.stride)
            
        # Glorot initialization.
        nn.init.xavier_uniform_(self.real_convt.weight)
        nn.init.xavier_uniform_(self.im_convt.weight)
        
        
    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]
        
        ct_real = self.real_convt(x_real) - self.im_convt(x_im)
        ct_im = self.im_convt(x_real) + self.real_convt(x_im)
        
        output = torch.stack([ct_real, ct_im], dim=-1)
        return output

class CBatchNorm2d(nn.Module):
    """
    Class of complex valued batch normalization layer
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        self.real_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                      affine=self.affine, track_running_stats=self.track_running_stats)
        self.im_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum,
                                    affine=self.affine, track_running_stats=self.track_running_stats) 
        
    def forward(self, x):
        x_real = x[..., 0]
        x_im = x[..., 1]
        
        n_real = self.real_b(x_real)
        n_im = self.im_b(x_im)  
        
        output = torch.stack([n_real, n_im], dim=-1)
        return output

class Encoder(nn.Module):
    """
    Class of upsample block
    """
    def __init__(self, filter_size=(7,5), stride_size=(2,2), in_channels=1, out_channels=45, padding=(0,0)):
        super().__init__()
        
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        self.cconv = CConv2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                             kernel_size=self.filter_size, stride=self.stride_size, padding=self.padding)
        
        self.cbn = CBatchNorm2d(num_features=self.out_channels) 
        
        self.leaky_relu = nn.LeakyReLU()
            
    def forward(self, x):
        
        conved = self.cconv(x)
        normed = self.cbn(conved)
        acted = self.leaky_relu(normed)
        
        return acted

class Decoder(nn.Module):
    """
    Class of downsample block
    """
    def __init__(self, filter_size=(7,5), stride_size=(2,2), in_channels=1, out_channels=45,
                 output_padding=(0,0), padding=(0,0), last_layer=False):
        super().__init__()
        
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_padding = output_padding
        self.padding = padding
        
        self.last_layer = last_layer
        
        self.cconvt = CConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                             kernel_size=self.filter_size, stride=self.stride_size, output_padding=self.output_padding, padding=self.padding)
        
        self.cbn = CBatchNorm2d(num_features=self.out_channels) 
        
        self.leaky_relu = nn.LeakyReLU()
            
    def forward(self, x):
        
        conved = self.cconvt(x)
        
        if not self.last_layer:
            normed = self.cbn(conved)
            output = self.leaky_relu(normed)
        else:
            m_phase = conved / (torch.abs(conved) + 1e-8)
            m_mag = torch.tanh(torch.abs(conved))
            output = m_phase * m_mag
            
        return output

class DCUnet10(nn.Module):
    """
    Deep Complex U-Net.
    """
    def __init__(self, n_fft, hop_length):
        super().__init__()
        
        # for istft
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # downsampling/encoding
        self.downsample0 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=1, out_channels=45)
        self.downsample1 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=45, out_channels=90)
        self.downsample2 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=90, out_channels=90)
        self.downsample3 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=90, out_channels=90)
        self.downsample4 = Encoder(filter_size=(3,3), stride_size=(2,1), in_channels=90, out_channels=90)
        
        # upsampling/decoding
        self.upsample0 = Decoder(filter_size=(3,3), stride_size=(2,1), in_channels=90, out_channels=90)
        self.upsample1 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=180, out_channels=90)
        self.upsample2 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=180, out_channels=90)
        self.upsample3 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=180, out_channels=45)
        self.upsample4 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=90, output_padding=(1,1),
                                 out_channels=1, last_layer=True)
        
    def forward(self, x, n_fft, hop_length, is_istft=True):
        """Memory-optimized forward pass with input validation."""
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor input, got {type(x)}")
            
        if x.dim() != 5 or x.size(-1) != 2:
            raise ValueError(f"Expected input shape [B, C, F, T, 2], got {x.shape}")
            
        if x.numel() == 0:
            raise ValueError("Empty input tensor")
            
        # Store original input for residual connection (more memory efficient than keeping all intermediates)
        x_orig = x
        
        # Encoder path - downsample
        d0 = self.downsample0(x)
        d1 = self.downsample1(d0) 
        d2 = self.downsample2(d1)       
        d3 = self.downsample3(d2)    
        d4 = self.downsample4(d3)
        
        # Decoder path with skip connections
        u0 = self.upsample0(d4)
        # Use in-place operations where possible to save memory
        c0 = torch.cat((u0, d3), dim=1)
        del d3  # Free memory as soon as possible
        
        u1 = self.upsample1(c0)
        del c0
        c1 = torch.cat((u1, d2), dim=1)
        del d2
        
        u2 = self.upsample2(c1) 
        del c1
        c2 = torch.cat((u2, d1), dim=1)
        del d1
        
        u3 = self.upsample3(c2)
        del c2
        c3 = torch.cat((u3, d0), dim=1)
        del d0
        
        u4 = self.upsample4(c3)
        del c3

        # Apply mask to original input
        output = u4 * x_orig
        del u4  # Free mask tensor

        if is_istft:
            try:
                output = torch.squeeze(output, 1)
                if output.size(-1) != 2:
                    raise ValueError(f"Expected real/imag format with last dim=2, got {output.shape}")
                    
                # Convert from real/imaginary format to complex tensor
                output_complex = torch.complex(output[..., 0], output[..., 1])
                
                # Create window tensor on correct device
                window = torch.hann_window(n_fft, device=output.device)
                output = torch.istft(output_complex, n_fft=n_fft, hop_length=hop_length, 
                                   normalized=True, window=window)
                                   
            except Exception as e:
                raise RuntimeError(f"ISTFT conversion failed: {str(e)}")
        
        return output

class DCUnet10_rTSTM(nn.Module):
    """
    Deep Complex U-Net with real TSTM.
    """
    def __init__(self, n_fft, hop_length):
        super().__init__()
        
        # for istft
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # downsampling/encoding
        self.downsample0 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=1, out_channels=32)
        self.downsample1 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=32, out_channels=64)
        self.downsample2 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=64, out_channels=64)
        self.downsample3 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=64, out_channels=64)
        self.downsample4 = Encoder(filter_size=(3,3), stride_size=(2,1), in_channels=64, out_channels=64)
        
        # TSTM
        self.dual_transformer = Dual_Transformer(64, 64, nhead=4, num_layers=6)   # [b, 64, nframes, 8]

        # upsampling/decoding
        self.upsample0 = Decoder(filter_size=(3,3), stride_size=(2,1), in_channels=64, out_channels=64)
        self.upsample1 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=128, out_channels=64)
        self.upsample2 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=128, out_channels=64)
        self.upsample3 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=128, out_channels=32)
        self.upsample4 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=64, output_padding=(1,1),
                                 out_channels=1, last_layer=True)
        
    def forward(self, x, n_fft, hop_length, is_istft=True):
        """Memory-optimized forward pass with real TSTM and input validation."""
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor input, got {type(x)}")
            
        if x.dim() != 5 or x.size(-1) != 2:
            raise ValueError(f"Expected input shape [B, C, F, T, 2], got {x.shape}")
            
        if x.numel() == 0:
            raise ValueError("Empty input tensor")
            
        # Store original for residual connection
        x_orig = x
        
        # Encoder path
        d0 = self.downsample0(x)
        d1 = self.downsample1(d0) 
        d2 = self.downsample2(d1)       
        d3 = self.downsample3(d2)    
        d4 = self.downsample4(d3)
        
        # Real TSTM processing - memory efficient approach
        d4_real = d4[..., 0]  # Real part
        d4_imag = d4[..., 1]  # Imaginary part
        
        # Process through transformer (sequential to save memory)
        d4_real_processed = self.dual_transformer(d4_real)
        d4_imag_processed = self.dual_transformer(d4_imag)
        
        # Reconstruct complex representation in-place
        d4[..., 0] = d4_real_processed
        d4[..., 1] = d4_imag_processed
        
        # Clean up intermediate tensors
        del d4_real, d4_imag, d4_real_processed, d4_imag_processed

        # Decoder path with memory cleanup
        u0 = self.upsample0(d4)
        c0 = torch.cat((u0, d3), dim=1)
        del d3
        
        u1 = self.upsample1(c0)
        del c0
        c1 = torch.cat((u1, d2), dim=1)
        del d2
        
        u2 = self.upsample2(c1)
        del c1
        c2 = torch.cat((u2, d1), dim=1)
        del d1
        
        u3 = self.upsample3(c2)
        del c2
        c3 = torch.cat((u3, d0), dim=1)
        del d0
        
        u4 = self.upsample4(c3)
        del c3

        # Apply mask
        output = u4 * x_orig
        del u4

        if is_istft:
            try:
                output = torch.squeeze(output, 1)
                if output.size(-1) != 2:
                    raise ValueError(f"Expected real/imag format, got {output.shape}")
                    
                # Create window on correct device
                window = torch.hann_window(n_fft, device=output.device)
                
                # Convert to complex and apply ISTFT
                output_complex = torch.complex(output[..., 0], output[..., 1])
                output = torch.istft(output_complex, n_fft=n_fft, hop_length=hop_length, 
                                   normalized=True, window=window)
                                   
            except Exception as e:
                raise RuntimeError(f"ISTFT conversion failed: {str(e)}")
        
        return output

class DCUnet10_cTSTM(nn.Module):
    """
    Deep Complex U-Net with complex TSTM.
    """
    def __init__(self, n_fft, hop_length):
        super().__init__()
        
        # for istft
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # downsampling/encoding
        self.downsample0 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=1, out_channels=32)
        self.downsample1 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=32, out_channels=64)
        self.downsample2 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=64, out_channels=64)
        self.downsample3 = Encoder(filter_size=(3,3), stride_size=(2,2), in_channels=64, out_channels=64)
        self.downsample4 = Encoder(filter_size=(3,3), stride_size=(2,1), in_channels=64, out_channels=64)
        
        # TSTM
        self.dual_transformer_real = Dual_Transformer(64, 64, nhead=4, num_layers=6)   # [b, 64, nframes, 8]
        self.dual_transformer_imag = Dual_Transformer(64, 64, nhead=4, num_layers=6)   # [b, 64, nframes, 8]

        # upsampling/decoding
        self.upsample0 = Decoder(filter_size=(3,3), stride_size=(2,1), in_channels=64, out_channels=64)
        self.upsample1 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=128, out_channels=64)
        self.upsample2 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=128, out_channels=64)
        self.upsample3 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=128, out_channels=32)
        self.upsample4 = Decoder(filter_size=(3,3), stride_size=(2,2), in_channels=64, output_padding=(1,1),
                                 out_channels=1, last_layer=True)
        
    def forward(self, x, n_fft, hop_length, is_istft=True):
        """Memory-optimized forward pass with complex TSTM and input validation."""
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor input, got {type(x)}")
            
        if x.dim() != 5 or x.size(-1) != 2:
            raise ValueError(f"Expected input shape [B, C, F, T, 2], got {x.shape}")
            
        if x.numel() == 0:
            raise ValueError("Empty input tensor")
            
        # Store original for residual connection
        x_orig = x
        
        # Encoder path
        d0 = self.downsample0(x)
        d1 = self.downsample1(d0) 
        d2 = self.downsample2(d1)       
        d3 = self.downsample3(d2)    
        d4 = self.downsample4(d3)
        
        # Complex TSTM processing - memory efficient
        d4_real = d4[..., 0]
        d4_imag = d4[..., 1]

        # Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        # Process components through transformers
        real_part_real = self.dual_transformer_real(d4_real)
        imag_part_imag = self.dual_transformer_imag(d4_imag) 
        imag_part_real = self.dual_transformer_imag(d4_real)
        real_part_imag = self.dual_transformer_real(d4_imag)
        
        # Compute complex TSTM output
        out_real = real_part_real - imag_part_imag
        out_imag = imag_part_real + real_part_imag
        
        # Clean up intermediate tensors
        del real_part_real, imag_part_imag, imag_part_real, real_part_imag
        del d4_real, d4_imag
        
        # Update d4 in-place to save memory
        d4[..., 0] = out_real
        d4[..., 1] = out_imag
        del out_real, out_imag

        # Decoder path with memory cleanup
        u0 = self.upsample0(d4)
        c0 = torch.cat((u0, d3), dim=1)
        del d3
        
        u1 = self.upsample1(c0)
        del c0
        c1 = torch.cat((u1, d2), dim=1)
        del d2
        
        u2 = self.upsample2(c1)
        del c1 
        c2 = torch.cat((u2, d1), dim=1)
        del d1
        
        u3 = self.upsample3(c2)
        del c2
        c3 = torch.cat((u3, d0), dim=1)
        del d0
        
        u4 = self.upsample4(c3)
        del c3

        # Apply mask
        output = u4 * x_orig
        del u4

        if is_istft:
            try:
                output = torch.squeeze(output, 1)
                if output.size(-1) != 2:
                    raise ValueError(f"Expected real/imag format, got {output.shape}")
                    
                # Create window on correct device
                window = torch.hann_window(n_fft, device=output.device)
                
                # Convert and apply ISTFT
                output_complex = torch.complex(output[..., 0], output[..., 1])
                output = torch.istft(output_complex, n_fft=n_fft, hop_length=hop_length, 
                                   normalized=True, window=window)
                                   
            except Exception as e:
                raise RuntimeError(f"ISTFT conversion failed: {str(e)}")
        
        return output