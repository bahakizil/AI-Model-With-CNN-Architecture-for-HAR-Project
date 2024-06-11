from PIL import Image
from torchvision import transforms

# Görüntüyü yükleme ve ön işleme
image = Image.open('path_to_your_image.jpg')
transform = transforms.Compose([
    transforms.Resize((224, 224)),       # Görüntü boyutunu değiştir
    transforms.ToTensor(),               # Görüntüyü PyTorch tensörüne dönüştür
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize et
])

# Dönüşümü uygula ve boyut ekle (batch size = 1)
input_tensor = transform(image).unsqueeze(0)

# Model ile tahmin yap
with torch.no_grad():
    output = model(input_tensor)

# Sonuçları yorumla
_, predicted = torch.max(output, 1)
print('Predicted:', predicted.item())
