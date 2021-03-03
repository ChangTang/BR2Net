# coding: utf-8
import os

datasets_root = '../DataSets/BlurDtection/CVPR2014/BlurTrainingTesting/'
Testroot= '../DataSets/BlurDtection/'

# For each dataset, I put images and masks together
cvpr2014_trainning_path = os.path.join(datasets_root, 'TrainingImgsGTExtended')
CVPR2014_path = os.path.join(Testroot, 'CVPR2014/BlurTrainingTesting/Imgs')
DUT_path = os.path.join(Testroot, 'DUT/Imgs')
CTCUG_path = os.path.join(Testroot, 'CTCUG/Imgs')

