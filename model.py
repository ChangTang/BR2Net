import torch
import torch.nn.functional as F
from torch import nn

from resnext import ResNeXt101


class BR2Net(nn.Module):
    def __init__(self):
        super(BR2Net, self).__init__()
        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        ## channel weighted feature maps
        self.CAlayer0 = nn.Sequential(CALayer(64,16))
        self.CAlayer1 = nn.Sequential(CALayer(256,16))
        self.CAlayer2 = nn.Sequential(CALayer(512,16))
        self.CAlayer3 = nn.Sequential(CALayer(1024,16))
        self.CAlayer4 = nn.Sequential(CALayer(2048,16))

        ## Low to High
        self.predictL2H0 = nn.Conv2d(64, 1, kernel_size=1)
        self.predictL2H1 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predictL2H2 = nn.Sequential(
            nn.Conv2d(513, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predictL2H3 = nn.Sequential(
            nn.Conv2d(1025, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predictL2H4 = nn.Sequential(
            nn.Conv2d(2049, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )

		## High to Low 
        self.predictH2L0 = nn.Conv2d(2048, 1, kernel_size=1)
        self.predictH2L1 = nn.Sequential(
            nn.Conv2d(1025, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predictH2L2 = nn.Sequential(
            nn.Conv2d(513, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predictH2L3 = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.predictH2L4 = nn.Sequential(
            nn.Conv2d(65, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )

		###
        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

	l0_size = layer0.size()[2:]
	l4_size = layer4.size()[2:]

        ## Compute CA weighted features
        CAlayer0 = self.CAlayer0(layer0)
        CAlayer1 = self.CAlayer1(layer1)
        CAlayer2 = self.CAlayer2(layer2)
        CAlayer3 = self.CAlayer3(layer3)
        CAlayer4 = self.CAlayer4(layer4)

	predictL2H0 = self.predictL2H0(CAlayer0)
        predictL2H1 = self.predictL2H1(torch.cat((predictL2H0, F.upsample(CAlayer1, size=l0_size, mode='bilinear')), 1)) + predictL2H0
        predictL2H2 = self.predictL2H2(torch.cat((predictL2H1, F.upsample(CAlayer2, size=l0_size, mode='bilinear')), 1)) + predictL2H1
	predictL2H3 = self.predictL2H3(torch.cat((predictL2H2, F.upsample(CAlayer3, size=l0_size, mode='bilinear')), 1)) + predictL2H2
	predictL2H4 = self.predictL2H4(torch.cat((predictL2H3, F.upsample(CAlayer4, size=l0_size, mode='bilinear')), 1)) + predictL2H3

	predictH2L0 = self.predictH2L0(CAlayer4)
        predictH2L1 = self.predictH2L1(torch.cat((predictH2L0, F.upsample(CAlayer3, size=l4_size, mode='bilinear')), 1)) + predictH2L0
        predictH2L2 = self.predictH2L2(torch.cat((predictH2L1, F.upsample(CAlayer2, size=l4_size, mode='bilinear')), 1)) + predictH2L1
	predictH2L3 = self.predictH2L3(torch.cat((predictH2L2, F.upsample(CAlayer1, size=l4_size, mode='bilinear')), 1)) + predictH2L2
	predictH2L4 = self.predictH2L4(torch.cat((predictH2L3, F.upsample(CAlayer0, size=l4_size, mode='bilinear')), 1)) + predictH2L3

        

	predictL2H0 = F.upsample(predictL2H0, size=x.size()[2:], mode='bilinear')
	predictL2H1 = F.upsample(predictL2H1, size=x.size()[2:], mode='bilinear')
	predictL2H2 = F.upsample(predictL2H2, size=x.size()[2:], mode='bilinear')
	predictL2H3 = F.upsample(predictL2H3, size=x.size()[2:], mode='bilinear')
	predictL2H4 = F.upsample(predictL2H4, size=x.size()[2:], mode='bilinear')

	predictH2L0 = F.upsample(predictH2L0, size=x.size()[2:], mode='bilinear')
	predictH2L1 = F.upsample(predictH2L1, size=x.size()[2:], mode='bilinear')
	predictH2L2 = F.upsample(predictH2L2, size=x.size()[2:], mode='bilinear')
	predictH2L3 = F.upsample(predictH2L3, size=x.size()[2:], mode='bilinear')
	predictH2L4 = F.upsample(predictH2L4, size=x.size()[2:], mode='bilinear')

        predictFusion = (predictL2H4 + predictH2L4)/2
		
        if self.training:
            return predictL2H0, predictL2H1, predictL2H2, predictL2H3, predictL2H4, predictH2L0, predictH2L1, predictH2L2, predictH2L3, predictH2L4, predictFusion
        return F.sigmoid(predictFusion)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


