from PIL import Image, ImageFilter

# Görüntüyü yükleme
image = Image.open('path_to_your_image.jpg')

# Görüntüyü döndürme
rotated_image = image.rotate(45)  # 45 derece döndür

# Görüntüyü filtreleme
blurred_image = image.filter(ImageFilter.BLUR)

# Görüntü boyutunu değiştirme
resized_image = image.resize((300, 300))

# Görüntüyü kaydetme
resized_image.save('resized_image.jpg')
