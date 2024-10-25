import torch
import logging
from sympy.codegen.ast import float32
from torch import nn
import torch.nn.functional as F
from TCAN.tcn_block import TemporalConvNet

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s'
    )

class BTCNet(nn.Module):
    def __init__(self, emb_size, output_size, features_size, num_channels=[128, 128, 128, 128], 
                 num_sub_blocks=2, temp_attn=True, nheads=1, en_res=True, conv=True, key_size=300, 
                 kernel_size=2, dropout=0.3, emb_dropout=0.1, visual=True, dtype=float32, device='cuda'):
        super(BTCNet, self).__init__()
        self.temp_attn = temp_attn
        self.num_levels = len(num_channels)

        # Define two TemporalConvNets for forward and backward directions
        self.tcanet_forward = TemporalConvNet(emb_size, output_size, num_channels, 
                                              num_sub_blocks, temp_attn, nheads, en_res, conv, key_size, 
                                              kernel_size, visual=visual, dropout=dropout)
        
        self.tcanet_backward = TemporalConvNet(emb_size, output_size, num_channels, 
                                               num_sub_blocks, temp_attn, nheads, en_res, conv, key_size, 
                                               kernel_size, visual=visual, dropout=dropout)
        
        self.drop = nn.Dropout(emb_dropout)
        self.decoder = nn.Linear(num_channels[-1] * 2, output_size)  # Double the size for bidirectional output
        self.final_conv = nn.Conv1d(num_channels[-1] * 2, output_size, 1)  # Adjust to handle bidirectional channels
        self.init_weights()

    def init_weights(self):
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        # Forward direction
        if self.temp_attn:
            y_forward, attn_weight_list_fwd = self.tcanet_forward(input)
            y_backward, attn_weight_list_bwd = self.tcanet_backward(torch.flip(input, dims=[1]))  # Reverse the input
            
            # Concatenate forward and backward outputs
            y = torch.cat([y_forward, y_backward], dim=1)
            
            # Apply final convolution
            y = self.final_conv(y)
            return y
        else:
            y_forward = self.tcanet_forward(input.transpose(1, 2))
            y_backward = self.tcanet_backward(torch.flip(input.transpose(1, 2), dims=[1]))
            
            y = torch.cat([y_forward, y_backward], dim=1)
            y = self.decoder(y.transpose(1, 2))
            return y.contiguous()
