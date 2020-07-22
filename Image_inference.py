from PIL import Image
import os
import glob

import torch
import torchvision
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms

class ImageClassifier():

    def __init__(self, model_path=None):

        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = 'cpu'

        model_ft = models.inception_v3(pretrained=None)
        num_classes = 2

        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        self.input_size = 299

        model = model_ft

        self.model = model
        self.model = self.model.to(self.device)
        if model_path is not None:
            self.model_path = model_path
            # self.model = torch.load(model_path)
            model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu') ))
            self.model.eval()
    #
    # def getModel(self, model_path):
    #     model = torch.load(model_path)
    #     return model

    def image_inference(self, image_path):
        img = Image.open(image_path)
        img = transforms.Resize(self.input_size)(img)
        img = transforms.ToTensor()(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(self.device)
        result = self.model(img)
        result = nn.Softmax()(result)
        return result.cpu().detach().numpy()

if __name__ == "__main__":

    model_path = 'data/inception83.pth'
    classifier = ImageClassifier(model_path=model_path)
    # test_image_path = r'K:\OneDrive_NJIT\OneDrive - NJIT\Research\Resilience\data\boston\building_images2\hlvA4Deozc_zh2J-gDUVUg_-70.975509_42.372362_0_82.00.jpg'
    images = glob.glob('data/sample_images/*.jpg')
    print(f'Images counts: {len(images)}')
    for jpg in images:
        result = classifier.image_inference(jpg)
        basename = os.path.basename(jpg)
        label = np.argmax(result)
        print(f"{basename} | label: {label}")
