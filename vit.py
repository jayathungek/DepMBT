import timm
import torch
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# model_name = 'vit_base_patch16_224'
# print(timm.list_models('vit*patch16*'))
# exit()
model_name = 'vit_large_patch16_224'
model = timm.create_model(model_name, pretrained=True)
print(model)
exit()
model.eval()

config = resolve_data_config({}, model=model)
transform = create_transform(**config)

url, filename = ("https://fastly.picsum.photos/id/557/500/500.jpg?hmac=VgywrD7fKCGhxMWKpslwlZCKExdTWsKc6v_4j9UrozM", "dog.jpg")
urllib.request.urlretrieve(url, filename)
img = Image.open(filename).convert('RGB')
tensor = transform(img).unsqueeze(0)  # transform and add batch dimension

with torch.no_grad():
    out = model(tensor)
probabilities = torch.nn.functional.softmax(out[0], dim=0)
print(probabilities.shape)
# prints: torch.Size([1000])

# Get imagenet class mappings
url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
urllib.request.urlretrieve(url, filename)
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Print top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
# prints class names and probabilities like:
# [('Samoyed', 0.6425196528434753), ('Pomeranian', 0.04062102362513542), ('keeshond', 0.03186424449086189), ('white wolf', 0.01739676296710968), ('Eskimo dog', 0.011717947199940681)]
