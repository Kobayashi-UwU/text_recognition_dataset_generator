import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import numpy as np


def load_fonts(fonts_dir):
    """Load all font files from the specified directory."""
    fonts = []
    for font_file in os.listdir(fonts_dir):
        if font_file.endswith(".ttf") or font_file.endswith(".otf"):
            fonts.append(os.path.join(fonts_dir, font_file))
    return fonts


def generate_image(text, font_path, padding=10):
    """Generate an image with the given text and font, with a random height between 30 and 100."""
    height = random.randint(30, 100)
    font_size = height - 2 * padding
    font = ImageFont.truetype(font_path, font_size)
    # Calculate text size using textbbox
    text_bbox = font.getbbox(text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    image_width = text_width + 2 * padding
    image = Image.new("RGB", (image_width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    text_x = padding
    text_y = (height - text_height) / 2

    # Randomly choose a gray color between 0 (black) and 169 (gray)
    gray_value = random.randint(0, 169)
    text_color = (gray_value, gray_value, gray_value)

    draw.text((text_x, text_y), text, font=font, fill=text_color)
    return image


def save_image_and_label(image, text, output_dir, filename):
    """Save the image and return the image path."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_path = os.path.join(output_dir, filename)
    image.save(image_path)
    return image_path


def add_noise(image):
    """Add random noise to an image."""
    np_image = np.array(image)
    noise = np.random.randint(
        0, 50, (np_image.shape[0], np_image.shape[1], np_image.shape[2]), dtype="uint8"
    )
    np_image = np.clip(np_image + noise, 0, 255)
    return Image.fromarray(np_image)


def augment_image(image):
    """Apply random augmentations to the image."""
    augmentations = [
        lambda img: img.rotate(random.uniform(-15, 15)),  # Random rotation
        lambda img: img.crop((10, 10, img.width - 10, img.height - 10)).resize(
            img.size
        ),  # Random crop and resize
        lambda img: ImageEnhance.Brightness(img).enhance(
            random.uniform(0.7, 1.3)
        ),  # Change brightness
        lambda img: ImageEnhance.Contrast(img).enhance(
            random.uniform(0.7, 1.3)
        ),  # Change contrast
        lambda img: img.filter(
            ImageFilter.GaussianBlur(random.uniform(0, 2))
        ),  # Apply blur
        lambda img: add_noise(img),  # Add noise
    ]
    augmentation = random.choice(augmentations)
    return augmentation(image)


def generate_dataset(
    text_list_path,
    fonts_dir,
    train_output_dir,
    valid_output_dir,
    split_ratio=0.8,
    augment=True,
):
    """Generate dataset with images and labels."""
    with open(text_list_path, "r", encoding="utf-8") as f:
        texts = f.readlines()
    texts = [text.strip() for text in texts]

    random.shuffle(texts)
    split_index = int(len(texts) * split_ratio)
    train_texts = texts[:split_index]
    valid_texts = texts[split_index:]

    fonts = load_fonts(fonts_dir)

    def process_texts(texts, output_dir, augment):
        labels = []
        for i, text in enumerate(texts):
            font_path = random.choice(fonts)
            image = generate_image(text, font_path)
            filename = f"img{i+1}.jpg"
            save_image_and_label(image, text, output_dir, filename)
            labels.append(f"{filename} {text}")

            if augment:
                # Apply augmentations and save augmented images
                for j in range(3):  # Generate 3 augmented versions for each image
                    augmented_image = augment_image(image)
                    augmented_filename = f"img{i+1}_aug{j+1}.jpg"
                    save_image_and_label(
                        augmented_image, text, output_dir, augmented_filename
                    )
                    labels.append(f"{augmented_filename} {text}")

        return labels

    train_labels = process_texts(train_texts, train_output_dir, augment)
    valid_labels = process_texts(valid_texts, valid_output_dir, augment)

    with open(os.path.join(train_output_dir, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_labels))

    with open(os.path.join(valid_output_dir, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(valid_labels))


# Paths
text_list_path = "./text_list.txt"
fonts_dir = "./fonts"
train_output_dir = "./train"
valid_output_dir = "./valid"

# Generate dataset
generate_dataset(text_list_path, fonts_dir, train_output_dir, valid_output_dir)
