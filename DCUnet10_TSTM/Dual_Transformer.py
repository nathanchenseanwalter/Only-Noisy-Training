import torch.nn as nn
import torch
import numpy as np
from DCUnet10_TSTM.improve_single_trans import TransformerEncoderLayer
import os

class Dual_Transformer(nn.Module):
    """
    Deep duaL-path RNN.
    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        output_size: int, dimension of the output size.
        dropout: float, dropout ratio. Default is 0.
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, input_size, output_size, nhead=4, dropout=0, num_layers=1):
        super(Dual_Transformer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.input = nn.Sequential(
            nn.Conv2d(input_size, input_size // 2, kernel_size=1),
            nn.PReLU()
        )

        # dual-path RNN
        self.row_trans = nn.ModuleList([])
        self.col_trans = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(num_layers):
            self.row_trans.append(TransformerEncoderLayer(d_model=input_size // 2, nhead=nhead, dropout=dropout, bidirectional=True))
            self.col_trans.append(TransformerEncoderLayer(d_model=input_size // 2, nhead=nhead, dropout=dropout, bidirectional=True))
            self.row_norm.append(nn.GroupNorm(1, input_size // 2, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, input_size // 2, eps=1e-8))

        # output layer
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv2d(input_size // 2, output_size, 1)
                                    )

    def forward(self, input):
        """Memory-optimized dual-path transformer forward pass."""
        # Input validation
        if not isinstance(input, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(input)}")
            
        if input.dim() != 4:
            raise ValueError(f"Expected 4D input [b, c, num_frames, frame_size], got {input.dim()}D")
            
        if input.numel() == 0:
            raise ValueError("Empty input tensor")
            
        b, c, dim2, dim1 = input.shape
        
        # Initial projection
        output = self.input(input)
        
        # Dual-path processing with memory optimization
        for i in range(len(self.row_trans)):
            # Row processing - more memory efficient reshaping
            row_input = output.permute(3, 0, 2, 1).contiguous().view(dim1, b*dim2, -1)
            row_output = self.row_trans[i](row_input)
            
            # Avoid storing intermediate tensors
            row_output = row_output.view(dim1, b, dim2, -1).permute(1, 3, 2, 0).contiguous()
            row_output = self.row_norm[i](row_output)
            
            # In-place addition to save memory
            output.add_(row_output)
            del row_output  # Explicit cleanup

            # Column processing
            col_input = output.permute(2, 0, 3, 1).contiguous().view(dim2, b*dim1, -1)
            col_output = self.col_trans[i](col_input)
            
            col_output = col_output.view(dim2, b, dim1, -1).permute(1, 3, 0, 2).contiguous()
            col_output = self.col_norm[i](col_output)
            
            # In-place addition
            output.add_(col_output)
            del col_output  # Explicit cleanup

        # Final projection
        output = self.output(output)
        
        return output
