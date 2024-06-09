import torchvision
from torchvision import transforms
def Transform_Select(args):
    if args.model_select =='VGGFace2':
        Input = 224
        transform = transforms.Compose([torchvision.transforms.Resize(Input),
                                        transforms.ToTensor()])
                                        # transforms.Normalize((0.5, 0.5, 0.5),
                                        #                      (0.5, 0.5, 0.5))])
    elif args.model_select == 'Light_CNN_9' or args.model_select == 'Light_CNN_29' or args.model_select == 'Light_CNN_29_v2':
        Input = 128
        transform = transforms.Compose([torchvision.transforms.Resize(Input),
                                        transforms.Grayscale(num_output_channels=1),
                                        transforms.ToTensor()])
                                        # transforms.Normalize([0.5],[0.5])
    elif args.model_select == 'IR-50':
        Input = 112
        transform = transforms.Compose([torchvision.transforms.Resize(Input),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),
                                                             (0.5, 0.5, 0.5))])
    return transform
