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


	predictL2H0 = self.predictL2H0(layer0)
        predictL2H1 = self.predictL2H1(torch.cat((predictL2H0, F.upsample(layer1, size=l0_size, mode='bilinear')), 1)) + predictL2H0
        predictL2H2 = self.predictL2H2(torch.cat((predictL2H1, F.upsample(layer2, size=l0_size, mode='bilinear')), 1)) + predictL2H1
	predictL2H3 = self.predictL2H3(torch.cat((predictL2H2, F.upsample(layer3, size=l0_size, mode='bilinear')), 1)) + predictL2H2
	predictL2H4 = self.predictL2H4(torch.cat((predictL2H3, F.upsample(layer4, size=l0_size, mode='bilinear')), 1)) + predictL2H3

	predictH2L0 = self.predictH2L0(layer4)
        predictH2L1 = self.predictH2L1(torch.cat((predictH2L0, F.upsample(layer3, size=l4_size, mode='bilinear')), 1)) + predictH2L0
        predictH2L2 = self.predictH2L2(torch.cat((predictH2L1, F.upsample(layer2, size=l4_size, mode='bilinear')), 1)) + predictH2L1
	predictH2L3 = self.predictH2L3(torch.cat((predictH2L2, F.upsample(layer1, size=l4_size, mode='bilinear')), 1)) + predictH2L2
	predictH2L4 = self.predictH2L4(torch.cat((predictH2L3, F.upsample(layer0, size=l4_size, mode='bilinear')), 1)) + predictH2L3

        

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
        return F.sigmoid(predictL2H4)


