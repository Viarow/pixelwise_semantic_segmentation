import torch
import torch.optim as optim
import torch.nn as nn

from PIL import Image
from visualize_plt import Window

import torchvision
from torchvision.transforms import Compose, CenterCrop, Normalize
from torchvision.transforms import ToTensor, ToPILImage
from transforms import Colorize, Relabel, Tolabel

from FCN import FCN8, FCN16, FCN32
from UNet import UNet
from SegNet import SegNet

from argparse import ArgumentParser

input_transform = Compose([
    CenterCrop(256),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225])
])

target_transform = Compose([
    CenterCrop(256),
    Tolabel(),
    Relabel(255,21)
])

CLASSES_NUM = 22
color_transform = Colorize(CLASSES_NUM)
image_transform = ToPILImage()


trainset = torchvision.datasets.VOCSegmentation(root='./data', year='2012',
                                                image_set='train', download=True,
                                                transform=input_transform,target_transform=target_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)


testset = torchvision.datasets.VOCSegmentation(root='./data', year='2012',
                                                image_set='val', download=True,
                                                transform=input_transform,target_transform=target_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffe=True, num_workers=2)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(args, model):

    model.train()
    criterion = nn.CrossEntropyLoss()

    if args.model.startwith('FCN'):
        optimizer = optim.SGD(model.parameters(), 1e-4, .9, 2e-5)
    elif args.model == 'SegNet':
        optimizer = optim.SGD(model.parameters(), 1e-3, .9)
    elif args.model == 'UNet':
        optimizer = optim.Adam(model.parameters())

    window = Window()

    for epoch in range(0, args.epochs):
        losses = []

        for i,data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

            # show the image
            if args.image_period>0 and (i+1) % args.image_period == 0:
                image = inputs[0].cpu().data
                image[0] = image[0] * .229 + .485
                image[1] = image[1] * .224 + .456
                image[2] = image[2] * .225 + .406

                window.show_image(image, 'input( {}-{} )'.format(epoch+1,i+1))
                color_output = color_transform(outputs[0].cpu().max(0)[1].data)
                window.show_image(color_output, 'output( {}-{} )'.format(epoch+1,i+1))
                color_target = color_transform(labels[0].cpu.data)
                window.show_image(color_target, 'target( {}-{} )'.format(epoch+1,i+1))

            # show the average running loss
            if (i+1) % args.loss_period == 0:
                print('[%d, %5d] loss: %.3f' % (epoch, i+1, sum(losses)/len(losses)))

        # save the model state
        if (epoch+1) % args.save_period == 0:
            filename = '{}-{}.pth'.format(args.model, epoch+1)
            torch.save(model.state_dict(), filename)
            print('{} saved'.format(filename))



def evaluate(args, model):

    model.eval()

    input = input_transform(Image.open(args.eval_image))

    with torch.no_grad:
        output = model(input)
    # output =  model(Variable(input, volatile=True).unsqueeze(0))

    color_output = color_transform(output[0].data.max(0)[1])
    window = Window()
    window.show_image(color_output)

    #image_transform(label).save(args.label)

def main(args):
    Net = None
    if args.model == 'FCN8':
        Net = FCN8
    elif args.model == 'FCN16':
        Net = FCN16
    elif args.model == 'FCN32':
        Net = FCN32
    elif args.model == 'UNet':
        Net = UNet
    elif args.model == 'SegNet':
        Net = SegNet

    assert Net is not None, 'model {} is not available'.format(args.model)

    model = Net(CLASSES_NUM)
    model = model.to(device)

    if args.mode == 'train':
        train(args, model)
    elif args.mode == 'eval':
        evaluate(args, model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model', require=True)

    subparsers = parser.add_subparsers(dest='mode')
    subparsers.required = True

    evalParser = subparsers.add_parser('eval')
    evalParser.add_argument('eval_image')

    trainParser = subparsers.add_parser('train')
    trainParser.add_argument('epochs', type=int, default=50)
    trainParser.add_argument('image_period', type=int, default=0)
    trainParser.add_argument('loss_period', type=int, default=50)
    trainParser.add_argument('save_period', type=int, default=10)

    main(parser.parse_args())









