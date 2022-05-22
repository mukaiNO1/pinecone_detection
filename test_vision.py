import torchvision


model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)


print(model)