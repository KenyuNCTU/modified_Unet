import torch
import torch.nn.functional as F
from torch import nn, Tensor





class Hswish(nn.Module):
    def forward(self, x):
        swish = F.relu6(x + 3 , inplace = True)
        return x* swish/6.

class Hsigmoid(nn.Module):

    def forward(self, x):
        return F.relu6(x + 3, inplace = True)/6.

class SElayer(nn.Module):
    def __init__(self, inplanes, ratio = 0.25):
        super(SElayer, self).__init__()
        hidden_dim = int(inplanes*ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(inplanes, hidden_dim, bias = False)
        self.fc2 = nn.Linear(hidden_dim, inplanes, bias = False)
        self.activate = Hsigmoid()
        

    def forward(self, x):
        out = self.avg_pool(x).view(x.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.activate(out)
        out = out.unsqueeze(2).unsqueeze(3)
        out = x * out.expand_as(x)

        return out

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            Hswish(),
            SElayer(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            Hswish()
        )
        self.fit = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        residual = self.fit(x)
        x = self.conv(x)
        out = residual+x
        return out


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, upsample=True):
        super(up, self).__init__()

        if upsample:
            self.up = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        #print(x1.shape, x2.shape)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        #print(diffX, diffY)
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        if diffX == 1 :
            #print(1)
            x2 = F.pad(x2, (0,0,0,1))
        if diffY == 1 :
            #print(2)
            x2 = F.pad(x2, (1,0))
        if diffX == -1:
            #print(3)
           # print(x1.shape)
            x1 = F.pad(x1, (0,0,0,1))
            #print(x1.shape)
        if diffY == -1:
            #print(4)
            x1 = F.pad(x1, (1,0))
        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class ASPP_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernal_size, padding, dilation):
        super(ASPP_conv, self).__init__()
        self.dconv = nn.Conv2d(in_ch, out_ch, kernal_size, 1, padding, dilation)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = Hswish()

    def forward(self, x):
        x = self.dconv(x)
        x = self.bn(x)

        return self.act(x)




class UNet(nn.Module):
    def __init__(self, n_classes, n_channels=1, n_filters=32, upsample=True):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, n_filters)
        self.down1 = down(n_filters, n_filters*2)
        self.down2 = down(n_filters*2, n_filters*4)
        self.down3 = down(n_filters*4, n_filters*8)
        self.down4 = down(n_filters*8, n_filters*8)
        self.up1 = up(n_filters*16, n_filters*4, upsample=upsample)
        self.up2 = up(n_filters*8, n_filters*2, upsample=upsample)
        self.up3 = up(n_filters*4, n_filters, upsample=upsample)
        self.up4 = up(n_filters*2, n_filters, upsample=upsample)
        self.finaldrop = nn.Dropout2d(p=0.5)
        self.outc = outconv(n_filters, n_classes)
        self.conv = nn.Conv2d(n_filters*8, n_filters*4, 1)
        self.aspp1 = ASPP_conv(n_filters*8, n_filters*4, 1, padding = 0, dilation = 1)
        self.aspp2 = ASPP_conv(n_filters*8, n_filters*4, 3, padding = 2, dilation = 2)
        self.aspp3 = ASPP_conv(n_filters*8, n_filters*4, 3, padding = 4, dilation = 4)
        self.aspp4 = ASPP_conv(n_filters*8, n_filters*4, 3, padding = 8, dilation = 8)
        self.to_feature = ASPP_conv(n_filters*16, n_filters*4, 1, 0, 1)
        self.init_weights()


    def init_weights(layer):
        if isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight)
                layer.bias.data.fill_(0.01)
        elif isinstance(layer, nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight)

        elif isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x4 = self.conv(x4)
        x5 = self.conv(x5)
        x5 = F.interpolate(x5, x4.size()[2:], mode = 'bilinear', align_corners = True)
        #print(x4.shape, x5.shape)
        f = torch.cat([x4, x5], dim = 1)
        f1 = self.aspp1(f)
        f2 = self.aspp2(f)
        f3 = self.aspp3(f)
        f4 = self.aspp4(f)
        f = torch.cat([f1, f2, f3, f4], dim = 1)

        x = self.to_feature(f)
        
        #x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.finaldrop(x)
        x = self.outc(x)
        return x

def RU_net(classes):
    return UNet(classes)

if __name__ == "__main__":
    import torch
    model = RU_net(3)
    input = torch.rand(1,1, 128,256)
    output = model(input)
    print(output.size())