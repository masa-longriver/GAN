import torch
import torch.nn as nn

def get_outsize(height, width, padding, kernel, stride):
    new_height = ((height + 2 * padding - kernel) / stride) + 1
    new_width  = ((width  + 2 * padding - kernel) / stride) + 1
    
    return int(new_height), int(new_width)

def calc_outsize(c_data, c_model):
    height1, width1 = get_outsize(c_data['height'],
                                  c_data['width'],
                                  c_model['padding1'],
                                  c_model['kernel'],
                                  c_model['stride1'])
    height2, width2 = get_outsize(height1,
                                  width1,
                                  c_model['padding1'],
                                  c_model['kernel'],
                                  c_model['stride1'])
    height3, width3 = get_outsize(height2,
                                  width2,
                                  c_model['padding1'],
                                  c_model['kernel'],
                                  c_model['stride1'])
    height4, width4 = get_outsize(height3,
                                  width3,
                                  c_model['padding1'],
                                  c_model['kernel'],
                                  c_model['stride1'])
    
    return height4, width4


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.c_data  = config['data']
        self.c_model = config['model']
        out_height, out_width = calc_outsize(self.c_data, self.c_model)

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(self.c_model['noise_channel'],
                               self.c_model['channel4'],
                               kernel_size=(out_height, out_width),
                               stride=self.c_model['stride2'],
                               padding=self.c_model['padding2'],
                               bias=False),
            nn.BatchNorm2d(self.c_model['channel4']),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.c_model['channel4'],
                               self.c_model['channel3'],
                               kernel_size=self.c_model['kernel'],
                               stride=self.c_model['stride1'],
                               padding=self.c_model['padding1'],
                               bias=False),
            nn.BatchNorm2d(self.c_model['channel3']),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.c_model['channel3'],
                               self.c_model['channel2'],
                               kernel_size=self.c_model['kernel'],
                               stride=self.c_model['stride1'],
                               padding=self.c_model['padding1'],
                               bias=False),
            nn.BatchNorm2d(self.c_model['channel2']),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.c_model['channel2'],
                               self.c_model['channel1'],
                               kernel_size=self.c_model['kernel'],
                               stride=self.c_model['stride1'],
                               padding=self.c_model['padding1'],
                               bias=False),
            nn.BatchNorm2d(self.c_model['channel1']),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.c_model['channel1'],
                               self.c_data['channel'],
                               kernel_size=self.c_model['kernel'],
                               stride=self.c_model['stride1'],
                               padding=self.c_model['padding1'],
                               bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.c_data  = config['data']
        self.c_model = config['model']
        out_height, out_width = calc_outsize(self.c_data, self.c_model)

        self.conv = nn.Sequential(
            nn.Conv2d(self.c_data['channel'],
                      self.c_model['channel1'],
                      kernel_size=self.c_model['kernel'],
                      stride=self.c_model['stride1'],
                      padding=self.c_model['padding1'],
                      bias=False),
            nn.LeakyReLU(self.c_model['neg_slope'],
                         inplace=True),
            nn.Conv2d(self.c_model['channel1'],
                      self.c_model['channel2'],
                      kernel_size=self.c_model['kernel'],
                      stride=self.c_model['stride1'],
                      padding=self.c_model['padding1'],
                      bias=False),
            nn.BatchNorm2d(self.c_model['channel2']),
            nn.LeakyReLU(self.c_model['neg_slope'],
                         inplace=True),
            nn.Conv2d(self.c_model['channel2'],
                      self.c_model['channel3'],
                      kernel_size=self.c_model['kernel'],
                      stride=self.c_model['stride1'],
                      padding=self.c_model['padding1'],
                      bias=False),
            nn.BatchNorm2d(self.c_model['channel3']),
            nn.LeakyReLU(self.c_model['neg_slope'],
                         inplace=True),
            nn.Conv2d(self.c_model['channel3'],
                      self.c_model['channel4'],
                      kernel_size=self.c_model['kernel'],
                      stride=self.c_model['stride1'],
                      padding=self.c_model['padding1'],
                      bias=False),
            nn.BatchNorm2d(self.c_model['channel4']),
            nn.LeakyReLU(self.c_model['neg_slope'],
                         inplace=True),
            nn.Conv2d(self.c_model['channel4'],
                      1,
                      kernel_size=(out_height, out_width),
                      stride=self.c_model['stride2'],
                      padding=self.c_model['padding2'],
                      bias=False),
            nn.Sigmoid(),
            nn.Flatten(),
        )
    
    def forward(self, x):
        
        return self.conv(x)