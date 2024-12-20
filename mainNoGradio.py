import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

from tqdm import tqdm

import gradio as gr


NUM_STEPS = 1000
ALPHA = 1
BETA = 1e5


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gramMatrix(input):
    (b, ch, h, w) = input.size()
    features = input.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = gramMatrix(target).detach()

    def forward(self, input):
        G = gramMatrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


normalizationMean = [0.485, 0.456, 0.406]
normalizationStd = [0.229, 0.224, 0.225]


class Normalization(nn.Module):
    def __init__(self, mean, std, device):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std


def getModelAndLosses(content, style, device):
    vgg = models.vgg16(weights="DEFAULT").features.to(device).eval()

    normalization = Normalization(normalizationMean, normalizationStd, device).to(
        device
    )

    contentLayers = ["relu_1", "relu_2", "relu_3", "relu_4"]
    styleLayers = ["relu_1", "relu_2", "relu_3", "relu_4"]

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


def runStyleTransfer(
    content, style, target, numSteps, contentWeight, styleWeight, styleIndWeight, device
):
    target.requires_grad_(True)

    model, modelContentLosses, modelStyleLosses = getModelAndLosses(
        content, style, device
    )

    model.eval()
    model.requires_grad_(False)

    optimizer = optim.Adam([target], 1)

    for step in tqdm(range(1, numSteps + 1)):
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

        totalLoss = contentWeight * contentLoss + styleWeight * (
            styleIndWeight * styleLoss[0] + ((1 - styleIndWeight) * styleLoss[1])
        )

        totalLoss.backward()

        optimizer.step()

    with torch.no_grad():
        target.clamp_(0, 1)

    target.requires_grad_(False)

    return target


def importImages(contentImageLoc, styleImageLoc, splits, overlap, scale, device):
    contentImage = Image.open(contentImageLoc).convert("RGB")
    imageSizeW, imageSizeH = contentImage.size

    imageSizeW = int(imageSizeW * scale)
    imageSizeH = int(imageSizeH * scale)

    rescale = transforms.Resize((imageSizeH, imageSizeW))

    contentImage = rescale(contentImage)

    if splits != 0:
        splitWidth = (imageSizeW + (splits - 1) * overlap) // splits

        imageSizeH = (
            imageSizeH * (splitWidth * splits - overlap * (splits - 1))
        ) // imageSizeW
        imageSizeW = splitWidth * splits - overlap * (splits - 1)

    loader = transforms.Compose(
        [transforms.Resize((imageSizeH, imageSizeW)), transforms.ToTensor()]
    )

    contentImage = loader(contentImage).unsqueeze(0).to(device)

    styleImages = []
    for loc in styleImageLoc:
        image = Image.open(loc).convert("RGB")
        image = loader(image).unsqueeze(0).to(device)
        styleImages.append(image)

    print(contentImage.get_device())

    return contentImage, styleImages


def getSplitImages(contentImage, styleImages, splits, overlap):
    _, __, imageSizeH, imageSizeW = contentImage.size()

    if splits == 0:
        contentImage = contentImage.unsqueeze(0)
        styleImages = [style.unsqueeze(0) for style in styleImages]
        return contentImage, styleImages

    splitWidth = (imageSizeW + (splits - 1) * overlap) // splits

    contentImage = contentImage.unfold(3, splitWidth, splitWidth - overlap)
    contentImage = contentImage.movedim(3, 0)

    assert contentImage.size()[0] == splits, "Splitting failed"

    styleImages = [
        style.unfold(3, splitWidth, splitWidth - overlap).movedim(3, 0)
        for style in styleImages
    ]

    return contentImage, styleImages


def joinSplitImages(splitImages, overlap, device):
    targetImages = splitImages[0]

    for i in range(1, len(splitImages)):
        weights = torch.linspace(0, 1, overlap).to(device)
        weights = weights.view(1, 1, 1, overlap)

        targetImages[:, :, :, -overlap:] = torch.lerp(
            targetImages[:, :, :, -overlap:],
            splitImages[i][:, :, :, 0:overlap],
            weights,
        )

        targetImages = torch.cat(
            (targetImages, splitImages[i][:, :, :, overlap:]), dim=3
        )

    return targetImages


def runSplitStyleTransfer(
    contentLoc,
    styleLoc,
    numSteps,
    contentWeight,
    styleWeight,
    splits,
    overlap,
    scale,
    device,
):
    contentImage, styleImages = importImages(
        contentLoc, styleLoc, splits, overlap, scale, device
    )

    splitContentImages, splitStyleImages = getSplitImages(
        contentImage, styleImages, splits, overlap
    )

    targetImages = []

    for i, (content, style1, style2) in enumerate(
        zip(splitContentImages, splitStyleImages[0], splitStyleImages[1])
    ):
        print("Running split {}".format(i))
        targetImage = content.clone()
        targetImage = runStyleTransfer(
            content,
            [style1, style2],
            targetImage,
            numSteps,
            contentWeight,
            styleWeight,
            1 - i * (1 / (splits - 1)),
            device,
        )
        targetImages.append(targetImage)

    targetImage = joinSplitImages(targetImages, overlap, device)

    return contentImage, styleImages, targetImage


def style_transfer(
    contentImageLoc,
    styleImage1Loc,
    styleImage2Loc,
    num_steps=NUM_STEPS,
    alpha=ALPHA,
    beta=BETA,
):
    device = "cpu"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    contentImage, styleImages, targetImage = runSplitStyleTransfer(
        contentImageLoc,
        [styleImage1Loc, styleImage2Loc],
        num_steps,
        alpha,
        beta,
        splits=20,
        overlap=10,
        scale=1,
        device=device,
    )

    return targetImage


demo = gr.Interface(
    style_transfer,
    [
        gr.Image(label="Content Image", type="filepath"),
        gr.Image(label="Style Image 1", type="filepath"),
        gr.Image(label="Style Image 2", type="filepath"),
        gr.Number(NUM_STEPS, label="Number of Steps"),
        gr.Number(ALPHA, label="Content Weight"),
        gr.Number(BETA, label="Style Weight"),
    ],
    gr.Image(label="Output Image"),
    title="Style Transfer",
    description="Transfer the style of two images to a third image.",
)

demo.launch()
