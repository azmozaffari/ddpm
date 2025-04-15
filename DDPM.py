import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn 
import argparse
import yaml



# Time embedding function
def get_time_embedding(time_steps, temb_dim):

    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    
    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )
    
    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb




class LinearNoiseScheduler():
    def __init__(self, T, beta_start, beta_end, device): # It divides the interval [0,1] to T timesteps
        self.device = device
        self.beta = torch.linspace(beta_start, beta_end, steps=T).to(device)        
        self.alpha = 1-self.beta
        self.alpha = torch.cumprod(self.alpha, dim=0).to(device)
        
        self.one_beta_sqrt = torch.sqrt(1 - self.beta)
        self.beta_sqrt = torch.sqrt(self.beta)

        self.one_alpha_sqrt = torch.sqrt(1 - self.alpha).to(device)
        self.alpha_sqrt = torch.sqrt(self.alpha)

    def add_noise(self, original, epsilon, time_step ): #epsilon ~N(0,I) with the same dimentions as the original signal
        noisy_signal = self.alpha_sqrt[time_step][:,None, None, None] * original + self.one_alpha_sqrt[time_step][:,None, None, None] * epsilon

        return noisy_signal
    
    def get_apha_beta_timelist(self,):
        return self.alpha_sqrt, self.beta_sqrt, self.one_alpha_sqrt, self.one_beta_sqrt 


    def sampling_from_xt_1(self, x_t, noise_predicted, time_step): # sample from q(x_t_1|x_t,x_0)
        x_t_1 = (1/self.one_beta_sqrt[time_step])*(x_t-(self.beta[time_step]/self.one_alpha_sqrt[time_step])*noise_predicted) +\
              (self.one_alpha_sqrt[time_step-1]/self.one_alpha_sqrt[time_step]) * self.beta_sqrt[time_step] * (torch.normal(0, 1, size=x_t.size()).to(self.device))
        x_0 = (1/self.alpha_sqrt[time_step])*(x_t - self.one_alpha_sqrt[time_step] * noise_predicted)

        x_0 = torch.clamp(x_0, -1., 1.)
        
        return x_t_1, x_0





class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, t_emb_dim,
                 down_sample=True, num_heads=4, num_layers=1, group_norm=3, patch_size=4):
        super().__init__()

        
        self.num_layers = num_layers
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_heads = num_heads
        self.down_sample = down_sample
        self.group_norm = group_norm
        self.patch_size = patch_size

        self.res_conv_block_one = nn.ModuleList([
            nn.Sequential(nn.GroupNorm(self.group_norm, in_channel if i==0 else out_channel),
                nn.SiLU(),
                nn.Conv2d(in_channel if i==0 else out_channel, out_channel, kernel_size=3, stride=1, padding=1))
            for i in range(num_layers)
        ])
        


        self.res_conv_blok_tow = nn.ModuleList([
            nn.Sequential(nn.GroupNorm(self.group_norm, out_channel),
                nn.SiLU(),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1))
            for _ in range(num_layers)
        ])


        self.time_embd = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channel)
            )
            for _ in range(num_layers)
        ])


        self.attentionNorm = nn.ModuleList([
            nn.GroupNorm(self.group_norm, out_channel)
            for _ in range(num_layers)
            ])


        self.attention = nn.ModuleList([
            nn.MultiheadAttention(self.out_channel*patch_size*patch_size, num_heads, batch_first=True)
            for _ in range(num_layers)
            ])


        self.skip_line_channel_alighnment = nn.ModuleList([
            nn.Conv2d(in_channel if i == 0 else out_channel, out_channel, kernel_size=1)
            for i in range(num_layers)
            ])
        

        self.down_sample_conv = nn.Conv2d(out_channel, out_channel,
                                          4, 2, 1) if self.down_sample else nn.Identity()



    def forward(self, x, t_emb):
        assert x.size(2)%self.patch_size == 0, "The input h size should be dividable by patch_size"
        assert x.size(3)%self.patch_size == 0, "The input w size should be dividable by patch_size"
        in_resnet = x
        for i in range(self.num_layers):
            # Resnet Block
            out_resnet = in_resnet
            out_resnet = self.res_conv_block_one[i](out_resnet)
            out_resnet = self.time_embd[i](t_emb)[:, :, None, None] + out_resnet
            out_resnet = self.res_conv_blok_tow[i](out_resnet)
            skip_line_resnet = self.skip_line_channel_alighnment[i](in_resnet)
            out_resnet = out_resnet + skip_line_resnet

            # Attention Block            
            batch_size, channels, h, w = out_resnet.size()
            in_att = out_resnet.reshape(batch_size, channels,h*w)
            in_att = self.attentionNorm[i](in_att)
            in_att = in_att.transpose(1,2)
            in_att = in_att.reshape(batch_size, int(h*w/(self.patch_size*self.patch_size)), self.out_channel*self.patch_size*self.patch_size)
            in_att = self.attention[i](in_att, in_att, in_att)[0]
            in_att = in_att.transpose(1,2).reshape(batch_size, channels, h, w)
            in_resnet = out_resnet + in_att

        out = self.down_sample_conv(in_resnet)
        return out, in_resnet
            


class MidBlock(nn.Module):
    def __init__(self, in_channel, out_channel, t_emb_dim, num_heads=4, num_layers=1, group_norm=3, patch_size=4):
        super().__init__()        
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.group_norm = group_norm
        self.patch_size = patch_size
        
        
        
        self.res_conv_block_one = nn.ModuleList([
            nn.Sequential(nn.GroupNorm(self.group_norm, in_channel if i==0 else out_channel),
                nn.SiLU(),
                nn.Conv2d(in_channel if i==0 else out_channel, out_channel, kernel_size=3, stride=1, padding=1))
            for i in range(num_layers+1)
        ])
        


        self.res_conv_blok_tow = nn.ModuleList([
            nn.Sequential(nn.GroupNorm(self.group_norm, out_channel),
                nn.SiLU(),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1))
            for _ in range(num_layers+1)
        ])


        self.time_embd = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channel)
            )
            for _ in range(num_layers+1)
        ])


        self.attentionNorm = nn.ModuleList([
            nn.GroupNorm(self.group_norm, out_channel)
            for _ in range(num_layers)
            ])


        self.attention = nn.ModuleList([
            nn.MultiheadAttention(self.out_channel*self.patch_size*self.patch_size, num_heads, batch_first=True)
            for _ in range(num_layers)
            ])
        self.skip_line_channel_alighnment = nn.ModuleList([
            nn.Conv2d(in_channel if i == 0 else out_channel, out_channel, kernel_size=1)
            for i in range(num_layers+1)
            ])

    def forward(self, x, t_emb):
        assert x.size(2)%self.patch_size == 0, "The input h size should be dividable by patch_size"
        assert x.size(3)%self.patch_size == 0, "The input w size should be dividable by patch_size"

        input_res = x
        out_res = self.res_conv_block_one[0](input_res)
        out_res = out_res + self.time_embd[0](t_emb)[:, :, None, None]
        out_res = self.res_conv_blok_tow[0](out_res)
        out_res = out_res + self.skip_line_channel_alighnment[0](input_res)
    
        for i in range(self.num_layers):
            # attention
            batch, n_channel, h, w  = out_res.size()
            in_att = out_res.reshape(batch, n_channel,h*w)
            in_att = self.attentionNorm[i](in_att)
            in_att = in_att.transpose(1,2)
            in_att = in_att.reshape(batch, int(h*w/(self.patch_size*self.patch_size)),self.out_channel*self.patch_size*self.patch_size )
            in_att = self.attention[i](in_att, in_att, in_att)[0]
            in_att = in_att.transpose(1,2).reshape(batch, n_channel, h, w)
            
            
            
            in_res = out_res + in_att
            out_res = self.res_conv_block_one[i+1](in_res)
            out_res = out_res + self.time_embd[i+1](t_emb)[:, :, None, None]
            out_res = self.res_conv_blok_tow[i+1](out_res)
            out_res = out_res + self.skip_line_channel_alighnment[i+1](in_res)

        return out_res







class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel,t_emb_dim,  num_heads=4, num_layers=1, group_norm=3, patch_size=4 ):
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.group_norm = group_norm
        self.patch_size = patch_size
        
        
        
        
        
        self.res_conv_block_one = nn.ModuleList([
            nn.Sequential(nn.GroupNorm(self.group_norm, in_channel if i==0 else out_channel),
                nn.SiLU(),
                nn.Conv2d(in_channel if i==0 else out_channel, out_channel, kernel_size=3, stride=1, padding=1))
            for i in range(num_layers)
        ])
        


        self.res_conv_blok_tow = nn.ModuleList([
            nn.Sequential(nn.GroupNorm(self.group_norm, out_channel),
                nn.SiLU(),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1))
            for _ in range(num_layers)
        ])


        self.time_embd = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channel)
            )
            for _ in range(num_layers)
        ])


        self.attentionNorm = nn.ModuleList([
            nn.GroupNorm(self.group_norm, out_channel)
            for _ in range(num_layers)
            ])


        self.attention = nn.ModuleList([
            nn.MultiheadAttention(self.out_channel*patch_size*patch_size, num_heads, batch_first=True)
            for _ in range(num_layers)
            ])
        self.skip_line_channel_alighnment = nn.ModuleList([
            nn.Conv2d(in_channel if i == 0 else out_channel, out_channel, kernel_size=1)
            for i in range(num_layers)
            ])

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest') # regarding to the downscaling function in DownBlock class the scaling factor is selected equals to 2




    def forward(self, x, downblock_x, t_emb):

        up_input = self.upsample(x)
        in_resnet = torch.concat((up_input,downblock_x), dim=1)

        assert in_resnet.size(2)%self.patch_size == 0, "The input h size should be dividable by patch_size"
        assert in_resnet.size(3)%self.patch_size == 0, "The input w size should be dividable by patch_size"

        for i in range(self.num_layers):
            # Resnet Block
            out_resnet = in_resnet
            out_resnet = self.res_conv_block_one[i](out_resnet)      
            out_resnet = self.time_embd[i](t_emb)[:, :, None, None] + out_resnet
            out_resnet = self.res_conv_blok_tow[i](out_resnet)
            in_resnet = self.skip_line_channel_alighnment[i](in_resnet)
            out_resnet = out_resnet + in_resnet

            # Attention Block            
            batch_size, channels, h, w = out_resnet.size()
            in_att = out_resnet.reshape(batch_size, channels,h*w)
            in_att = self.attentionNorm[i](in_att)
            in_att = in_att.transpose(1,2)
            in_att = in_att.reshape(batch_size, int(h*w/(self.patch_size*self.patch_size)),self.out_channel*self.patch_size*self.patch_size)
            in_att = self.attention[i](in_att, in_att, in_att)[0]
            in_att = in_att.transpose(1,2).reshape(batch_size, channels, h, w)
            in_resnet = out_resnet + in_att

        return in_resnet

    



class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.DB = nn.ModuleList([            
            DownBlock( in_channel=config.model.Unet.DownBlock.in_channel[i], out_channel=config.model.Unet.DownBlock.out_channel[i],
                      t_emb_dim=config.model.Unet.DownBlock.t_emb_dim, down_sample=True, num_heads=config.model.Unet.DownBlock.num_heads[i],
                      num_layers=config.model.Unet.DownBlock.num_layers[i], group_norm= config.model.Unet.DownBlock.group_norm[i], patch_size=config.model.Unet.patch_size)
            
        for i in range(config.model.Unet.depth)])

        self.UB = nn.ModuleList([
                UpBlock( in_channel=config.model.Unet.UpBlock.in_channel[i], out_channel=config.model.Unet.UpBlock.out_channel[i],
                        t_emb_dim=config.model.Unet.UpBlock.t_emb_dim, num_heads=config.model.Unet.UpBlock.num_heads[i],
                          num_layers=config.model.Unet.UpBlock.num_layers[i], group_norm= config.model.Unet.UpBlock.group_norm[i], patch_size=config.model.Unet.patch_size)
            
        for i in range(config.model.Unet.depth)])

        self.MB = MidBlock( in_channel= config.model.Unet.MidBlock.in_channel, out_channel=config.model.Unet.MidBlock.out_channel,
                            t_emb_dim=config.model.Unet.MidBlock.t_emb_dim, num_heads=config.model.Unet.MidBlock.num_heads, 
                            num_layers=config.model.Unet.MidBlock.num_layers, group_norm=config.model.Unet.MidBlock.group_norm, patch_size=config.model.Unet.patch_size)

    def forward(self, x, t_emb):

        in_up = []
        out = x
        for i in range(self.config.model.Unet.depth):
            out , input_up = self.DB[i](out,t_emb)
            in_up.append(input_up)
            
        
        middle = self.MB(out, t_emb)
        
        up = self.UB[0](middle, in_up[self.config.model.Unet.depth-1], t_emb)
        for i  in range(1, self.config.model.Unet.depth):
            up = self.UB[i](up, in_up[self.config.model.Unet.depth-i-1], t_emb)
        
        return up






#         ----------------- Improved MultiHeadAttention Implementation -----------------------------
# Using MultiHeadAttention as shown in original DDPM implementation can cause CUDA memory overflow, especially with high-resolution inputs.
# To fix this, I've changed the input format by reshaping it. Normally, the input is shaped as h * w and channels are considered as features.
#  Instead, I've splited the input into P×P patches (like in Vision Transformers), 
#  so we have fewer tokens (h * w / p * p) but each one has more features (channels * p * p).
# This helps reduce memory usage while still keeping useful information.




































