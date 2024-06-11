from PIL import Image
from torchvision import transforms

# Görüntüyü yüklemek ve transform etmek için fonksiyon
def load_image(image_path):
    with Image.open(image_path) as img:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),               # Görüntü boyutunu değiştir
            transforms.ToTensor(),                       # Görüntüyü tensor'e çevir
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize et
        ])
        return transform(img)

# Örnek bir görüntü yükleme
image_tensor = load_image('path_to_your_image.jpg')
