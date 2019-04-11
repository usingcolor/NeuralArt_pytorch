from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
from tensorboardX import SummaryWriter
import copy
import click

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('log')


size = 512
loader = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
unloader = transforms.ToPILImage()

def image_loader(image_path):
    image = Image.open(image_path)
    image = loader(image).unsqueeze(0) #torch.unsqueeze(input, dim, out=None) → Tensor

    return image.to(device, torch.float)


style_path = ''
content_path = ''
output_path = ''

style_image = image_loader(style_path)
content_image = image_loader(content_path)
input_image = content_image.clone()

assert style_image.size() == content_image.size(), "input images size error"


class Content_loss(nn.Module):
    def __init__(self, target):
        super(Content_loss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a,b,c,d = input.size()
    feature = input.view(a*b, c*d)
    G = torch.mm(feature, feature.t())

    return G.div(a*b*c*d)


class Style_loss(nn.Module):
    def __init__(self, target):
        super(Style_loss, self).__init__()
        self.target = gram_matrix(target).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1,1,1)
        self.std = torch.tensor(std).view(-1,1,1)

    def forward(self, input):
        return (input-self.mean)/self.std



def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img,
                               content_img, content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    idx = i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i+=1
            name = "conv_{}".format(i)
        elif isinstance(layer, nn.ReLU):
            name = "relu_{}".format(i)
        elif isinstance(layer, nn.MaxPool2d):
            name = "pool_{}".format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = "bn_{}".format(i)
        else:
            raise RuntimeError("Unexpected layer:{}".format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = Content_loss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)
            idx = len(model)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = Style_loss(target_feature)
            model.add_module("styel_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)
            idx = len(model)

        # model = model[:idx+1]

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], Content_loss) or isinstance(model[i], Style_loss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run(cnn, normalization_mean, normalization_std, content_img, style_img, input_image, num_steps = 300, style_weight = 10e6, content_weight = 1):
    print('Building a model')

    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_image, content_image)

    optimizer = get_input_optimizer(input_image)

    print("Optimizing")

    iteration = [0]
    while iteration[0] <= num_steps:

        def closure():
            input_image.data.clamp_(0,1)

            optimizer.zero_grad()
            model(input_image)
            style_score = 0
            content_score = 0

            for i in style_losses:
                style_score+=i.loss
            for i in content_losses:
                content_score+=i.loss

            style_score = style_weight*style_score
            content_score = content_weight*content_score
            loss = style_score+content_score
            loss.backward()

            iteration[0]+= 1

            writer.add_scalar('data/style_loss', style_score, iteration[0])
            writer.add_scalar('data/content_loss', content_score, iteration[0])
            writer.add_scalar('data/loss', loss, iteration[0])

            return style_score+content_score

        optimizer.step(closure())
        image = unloader(input_image)
        writer.add_image(image, iteration[0])


    input_image = input_image.data.clamp_(0,1)

    return input_image


#
# @click.command("style transfer with pytorch")
# @click.argument("style_image_path")
# @click.argument("content_image_path")
# @click.argument("output_image_path")
# @click.option("--image_size", default = 256)
# @click.option("--num_steps", default = 300)
# @click.option("--weight_rate", defult = 1e6)
#
# def main(style_image_path, content_image_path, output_image_path, image_size = 256, num_steps = 300, weight_rate = 1e6):
#     style_path = style_image_path
#     content_path = content_image_path

output = run(cnn, cnn_normalization_mean, cnn_normalization_std, content_image, style_image, input_image)
output_image = unloader(output)


output_image.save(output_path)

#
# if __name__ == '__main__':
#     main()







