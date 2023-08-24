from typing import Callable

import torch
from PIL import Image
import PIL.ImageOps
import torchvision.transforms as T
from facenet_pytorch import MTCNN


class CropFace(Callable):
    def __init__(self, size, margin, post_process=False):
        self.size = size
        self.margin = margin
        self.mtcnn = MTCNN(image_size=size, margin=margin, post_process=post_process)
    
    def __call__(self, image: torch.Tensor):
        face = self.mtcnn(image)
        return face


if __name__ == "__main__":
    crop = CropFace(size=224, margin=60)
    test_img = "test2.png" 
    to_tensor = T.ToTensor()
    pil = Image.open(test_img)
    tensor = to_tensor(pil)

    maybe_face =crop(pil)
    image_t = T.Compose([
        T.ToPILImage(),
    ])
    image = image_t(maybe_face)
    image = PIL.ImageOps.invert(image)
    image.save('face.png', format='PNG')