import torch
from PIL import Image
import torchvision.transforms as T
from facenet_pytorch import MTCNN


if __name__ == "__main__":
    test_img = "test.png" 
    to_tensor = T.ToTensor()
    pil = Image.open(test_img)
    tensor = to_tensor(pil)

    mtcnn = MTCNN(image_size=224, margin=60, post_process=False)
    maybe_face = mtcnn(pil)
    image_t = T.Compose([
        T.ToPILImage(),
    ])
    image = image_t(maybe_face)
    print(maybe_face, maybe_face.shape)
    image.save('face.png', format='PNG')