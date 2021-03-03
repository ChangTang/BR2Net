import datetime
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import joint_transforms
from config import cvpr2014_trainning_path
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir
from model_CA import BR2Net
from torch.backends import cudnn

cudnn.benchmark = True

torch.manual_seed(2019)
torch.cuda.set_device(0)

ckpt_path = './ckpt'
exp_name = 'BR2Net'

args = {
    'iter_num': 6000,  #
    'train_batch_size': 8,
    'last_iter': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': ''
}

joint_transform = joint_transforms.Compose([
    joint_transforms.RandomCrop(300),
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(10)
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

train_set = ImageFolder(cvpr2014_trainning_path, joint_transform, img_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=12, shuffle=True)

criterion = nn.BCEWithLogitsLoss().cuda()
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')


def main():
    net = BR2Net().cuda().train()

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print 'training resumes from ' + args['snapshot']
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)


def train(net, optimizer):
    curr_iter = args['last_iter']
    while True:
        total_loss_record = AvgMeter()
	lossL2H0_record, lossL2H1_record, lossL2H2_record, lossL2H3_record, lossL2H4_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        lossH2L0_record, lossH2L1_record, lossH2L2_record, lossH2L3_record, lossH2L4_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']

            inputs, labels = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()

            optimizer.zero_grad()
            outputsL2H0, outputsL2H1, outputsL2H2, outputsL2H3, outputsL2H4, outputsH2L0, outputsH2L1, outputsH2L2, outputsH2L3, outputsH2L4, outputsFusion = net(inputs)
	    lossL2H0 = criterion(outputsL2H0, labels)
            lossL2H1 = criterion(outputsL2H1, labels)
            lossL2H2 = criterion(outputsL2H2, labels)
            lossL2H3 = criterion(outputsL2H3, labels)
            lossL2H4 = criterion(outputsL2H4, labels)

	    lossH2L0 = criterion(outputsH2L0, labels)
            lossH2L1 = criterion(outputsH2L1, labels)
            lossH2L2 = criterion(outputsH2L2, labels)
            lossH2L3 = criterion(outputsH2L3, labels)
            lossH2L4 = criterion(outputsH2L4, labels)

	    lossFusion = criterion(outputsFusion, labels)


            total_loss = lossFusion + lossL2H0 + lossL2H1 + lossL2H2 + lossL2H3 + lossL2H4 + lossH2L0 + lossH2L1 + lossH2L2 + lossH2L3 + lossH2L4
            total_loss.backward()
            optimizer.step()

            total_loss_record.update(total_loss.data[0], batch_size)
            lossL2H0_record.update(lossL2H0.data[0], batch_size)
            lossL2H1_record.update(lossL2H1.data[0], batch_size)
            lossL2H2_record.update(lossL2H2.data[0], batch_size)
            lossL2H3_record.update(lossL2H3.data[0], batch_size)
            lossL2H4_record.update(lossL2H4.data[0], batch_size)

            lossH2L0_record.update(lossH2L0.data[0], batch_size)
            lossH2L1_record.update(lossH2L1.data[0], batch_size)
            lossH2L2_record.update(lossH2L2.data[0], batch_size)
            lossH2L3_record.update(lossH2L3.data[0], batch_size)
            lossH2L4_record.update(lossH2L4.data[0], batch_size)

            curr_iter += 1

            log = '[iter %d], [total loss %.5f], [lossL2H0 %.5f], [lossL2H1 %.5f], [lossL2H2 %.5f], [lossL2H3 %.5f], [lossL2H4 %.5f]' \
                  '[lossH2L0 %.5f], [lossH2L1 %.5f], [lossH2L2 %.5f], [lossH2L3 %.5f], [lossH2L4 %.5f], [lr %.13f]' % \
                   (curr_iter, total_loss_record.avg, lossL2H0_record.avg, lossL2H1_record.avg, lossL2H2_record.avg,
                   lossL2H3_record.avg, lossL2H4_record.avg, lossH2L0_record.avg, lossH2L1_record.avg, lossH2L2_record.avg,
                   lossH2L3_record.avg, lossH2L4_record.avg,
                   optimizer.param_groups[1]['lr'])
	    logWrite = '%d %.5f %.5f %.5f %.5f %.5f %.5f]' \
                  '%.5f %.5f %.5f %.5f %.5f %.13f]' % \
                   (curr_iter, total_loss_record.avg, lossL2H0_record.avg, lossL2H1_record.avg, lossL2H2_record.avg,
                   lossL2H3_record.avg, lossL2H4_record.avg, lossH2L0_record.avg, lossH2L1_record.avg, lossH2L2_record.avg,
                   lossH2L3_record.avg, lossH2L4_record.avg,
                   optimizer.param_groups[1]['lr'])
            print log
            open(log_path, 'a').write(logWrite + '\n')

            if curr_iter == args['iter_num']:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, 'CA_%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_path, exp_name, 'CA_%d_optim.pth' % curr_iter))
                return


if __name__ == '__main__':
    main()
