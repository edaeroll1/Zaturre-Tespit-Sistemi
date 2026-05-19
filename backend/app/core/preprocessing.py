from torchvision import transforms
from .config import IMG_SIZE, MEAN, STD

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[MEAN], std=[STD]),
])
