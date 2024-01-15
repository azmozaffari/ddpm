import  torch
import torch.nn as nn



class UnetResi(nn.Modoul):
    def __init__(self):
        super().__init__

    

    def starting_block(self,x, in_ch, stride):
        x = nn.Conv2d(in_ch,in_ch*2, stride)(x)
        x = nn.BatchNorm2d(x.shape(1))(x)
        x = nn.ReLU(x)
        x = nn.Conv2d(in_ch*2, stride)(x)





    def middle_block(self, x):
    






    # def BN(self, x, active):
    #     n_ch = x.shape(1)
    #     x = nn.BatchNorm2d(n_ch)(x)
    #     if active:


    # # the block of BN and convoution 
    # def conv(self,x, input_ch, output_ch, filter, stride ):

    #     x = BN(x)
    #     m = nn.Conv2d(input_ch, output_ch, filter, stride)(x)
    #     return m






































# def bn_act(x, act=True):
#     x = keras.layers.BatchNormalization()(x)
#     if act == True:
#         x = keras.layers.Activation("relu")(x)
#     return x

# def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
#     conv = bn_act(x)
#     conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
#     return conv

# def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
#     conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
#     conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    
#     shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
#     shortcut = bn_act(shortcut, act=False)
    
#     output = keras.layers.Add()([conv, shortcut])
#     return output

# def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
#     res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
#     res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
    
#     shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
#     shortcut = bn_act(shortcut, act=False)
    
#     output = keras.layers.Add()([shortcut, res])
#     return output

# def upsample_concat_block(x, xskip):
#     u = keras.layers.UpSampling2D((2, 2))(x)
#     c = keras.layers.Concatenate()([u, xskip])
#     return c

    



