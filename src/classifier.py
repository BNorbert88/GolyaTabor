from PIL import Image
import torch
from torchvision import transforms
import torchvision.models as models


def run_classifier_imagenet():
    input_image = Image.open('../dog.jpg')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define transformations to get a 224x224 image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)
    # Put data to GPU
    input_batch.to(device)

    with open("../image_classes.txt") as f:
        labels = eval(f.read())

    resnet = models.resnet152(pretrained=True)
    if device != 'cpu':
        resnet.cuda()
    # Set evaluation mode for resnet
    resnet.eval()
    # Forward propagation
    output = resnet(input_batch)

    # Sort indicies to get top5 classes
    _, indices = torch.sort(output, descending=True)
    # Apply softmax
    percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
    # Get top 5 classes
    result = [(labels[index], percentage[index].item()) for index in [indices[0][idx].item() for idx in range(5)]]

    return result
