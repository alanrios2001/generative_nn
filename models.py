from torch import nn

#generator model
class Generator(nn.Module):
    def __init__(self, zdim=256, d_dim=64):
        super(Generator, self).__init__()
        self.zdim = zdim
        self.gen = nn.Sequential(
            self.make_gen_block(zdim, d_dim*64, 4, 1, 0),
            self.make_gen_block(d_dim*64, d_dim*32, 4, 2, 1),
            self.make_gen_block(d_dim*32, d_dim*16, 4, 2, 1),
            self.make_gen_block(d_dim*16, d_dim*8, 4, 2, 1),
            self.make_gen_block(d_dim*8, d_dim*4, 4, 2, 1),
            self.make_gen_block(d_dim*4, d_dim*2, 4, 2, 1),
            self.make_gen_block(d_dim*2, 3, 4, 2, 1, final_layer=True),

        )

    def make_gen_block(self, input_channels, output_channels, kernel_size, stride, padding, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, bias=False),
                nn.Tanh(),
            )

    def forward(self, noise):
        x = noise.view(len(noise), self.zdim, 1, 1)
        return self.gen(x)


#critic model
class Critic(nn.Module):
    def __init__(self, d_dim=16):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            self.make_crit_block(3, d_dim, 4, 2, 1),
            self.make_crit_block(d_dim, d_dim * 2, 4, 2, 1),
            self.make_crit_block(d_dim * 2, d_dim * 4, 4, 2, 1),
            self.make_crit_block(d_dim * 4, d_dim * 8, 4, 2, 1),
            self.make_crit_block(d_dim * 8, d_dim * 16, 4, 2, 1),
            nn.Conv2d(d_dim*16, 1, 4, 1, 0, bias=False),
        )

    def make_crit_block(self, input_channels, output_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(output_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, image):
        crit_pred = self.crit(image)
        return crit_pred.view(len(crit_pred), -1)