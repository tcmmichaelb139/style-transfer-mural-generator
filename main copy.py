import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

from tqdm import tqdm

NUM_STEPS = 300
LEARNING_RATE = 0.001
ALPHA = 1
BETA = 1e6

device = "mps"
torch.set_default_device(device)


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = self.gramMatrix(target.detach())

    def gramMatrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    def forward(self, input):
        G = self.gramMatrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


normalizationMean = [0.485, 0.456, 0.406]
normalizationStd = [0.229, 0.224, 0.225]


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std


def getModelAndLosses(content, style):
    vgg = models.vgg19(weights="DEFAULT").features.to(device).eval()

    normalization = Normalization(normalizationMean, normalizationStd).to(device)

    contentLayers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]
    styleLayers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

    contentLosses = []
    styleLosses = [[] for _ in range(len(style))]

    model = nn.Sequential(normalization)

    convNum = 0
    for layer in vgg.children():
        if isinstance(layer, nn.Conv2d):
            convNum += 1
            name = "conv_{}".format(convNum)
        elif isinstance(layer, nn.ReLU):
            name = "relu_{}".format(convNum)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = "pool_{}".format(convNum)
        elif isinstance(layer, nn.BatchNorm2d):
            name = "bn_{}".format(convNum)
        else:
            raise RuntimeError(
                "Unrecognized layer: {}".format(layer.__class__.__name__)
            )

        model.add_module(name, layer)

        if name in contentLayers:
            target = model(content).detach()
            contentLoss = ContentLoss(target)
            model.add_module("content_loss_{}".format(convNum), contentLoss)
            contentLosses.append(contentLoss)

        if name in styleLayers:
            for i, styl in enumerate(style):
                target = model(styl).detach()
                styleLoss = StyleLoss(target)
                model.add_module("style_loss_{}_{}".format(convNum, i), styleLoss)
                styleLosses[i].append(styleLoss)

    for convNum in range(len(model) - 1, -1, -1):
        if isinstance(model[convNum], ContentLoss) or isinstance(
            model[convNum], StyleLoss
        ):
            break

    model = model[: (convNum + 1)]

    return model, contentLosses, styleLosses


def runStyleTransfer(content, style, target, style1Weight=1):
    target.requires_grad_(True)

    model, modelContentLosses, modelStyleLosses = getModelAndLosses(content, style)

    model.eval()
    model.requires_grad_(False)

    optimizer = optim.Adam([target], LEARNING_RATE)

    for step in tqdm(range(NUM_STEPS)):
        with torch.no_grad():
            target.clamp_(0, 1)

        optimizer.zero_grad()

        model(target)

        contentLoss = 0
        styleLoss = [0 for _ in range(len(style))]

        for loss in modelContentLosses:
            contentLoss += loss.loss

        for i in range(len(style)):
            for loss in modelStyleLosses[i]:
                styleLoss[i] += loss.loss

        totalLoss = ALPHA * contentLoss + BETA * (
            style1Weight * styleLoss[0] + ((1 - style1Weight) * styleLoss[1])
        )
        totalLoss.backward()

        optimizer.step()

        if step % 100 == 0:
            print("run {}:".format(step))
            print(
                "Style Losses : {:4f} | {:4f} Content Loss: {:4f}".format(
                    styleLoss[0].item(), styleLoss[1].item(), contentLoss.item()
                )
            )
            print()

    target.requires_grad_(False)

    return target


def importImages(contentImageLoc, styleImageLoc):
    contentImage = Image.open(contentImageLoc)

    print(contentImage.shape)


contentLoc = "./tests/images/dancing.jpg"
styleLoc = ["./tests/images/picasso.jpg", "./tests/images/picasso2.jpg"]

importImages()

# IMAGESIZEW = 256
# IMAGESIZEH = 256
# FRAMES = 10

# loader = transforms.Compose(
#     [transforms.Resize((IMAGESIZEH, IMAGESIZEW)), transforms.ToTensor()]
# )


# def imageLoader(image_name):
#     image = Image.open(image_name)
#     image = loader(image).unsqueeze(0)
#     return image.to(device, torch.float)


# styleImage = [
#     imageLoader("./tests/images/picasso.jpg"),
#     imageLoader("./tests/images/picasso2.jpg"),
# ]
# contentImage = imageLoader("./tests/images/dancing.jpg")
# targetImage = contentImage.clone()

# assert styleImage[0].size() == contentImage.size(), "Image sizes need to be the same"
# assert styleImage[1].size() == contentImage.size(), "Image sizes need to be the same"

# unloader = transforms.ToPILImage()

# targetImage = runStyleTransfer(contentImage, styleImage, targetImage, 1)


# fig, ax = plt.subplots(1, 4, figsize=(15, 5))
# ax[0].imshow(unloader(styleImage[0].squeeze(0)))
# ax[0].set_title("Style Image")
# ax[0].axis("off")
# ax[1].imshow(unloader(styleImage[1].squeeze(0)))
# ax[1].set_title("Style Image 2")
# ax[1].axis("off")
# ax[2].imshow(unloader(contentImage.squeeze(0)))
# ax[2].set_title("Content Image")
# ax[2].axis("off")
# ax[3].imshow(unloader(targetImage.squeeze(0)))
# ax[3].set_title("Output Image")
# ax[3].axis("off")

# plt.ioff()
# plt.show()
