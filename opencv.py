import cv2
import numpy as np
import matplotlib.pyplot as plt

def denoise_image(image_path):
    noise_image = cv2.imread(image_path)
    noise_image_rgb = cv2.cvtColor(noise_image, cv2.COLOR_BGR2RGB)
    
    gaussian_denoised_img = cv2.GaussianBlur(noise_image, (5, 5), 0)
    gaussian_denoised_img_rgb = cv2.cvtColor(gaussian_denoised_img, cv2.COLOR_BGR2RGB)

    median_denoised_img = cv2.medianBlur(noise_image, 5)
    median_denoised_img_rgb = cv2.cvtColor(median_denoised_img, cv2.COLOR_BGR2RGB)

    bilateral_denoised_img = cv2.bilateralFilter(noise_image, 9, 75, 75)
    bilateral_denoised_img_rgb = cv2.cvtColor(bilateral_denoised_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 10))

    # Original Image
    plt.subplot(2, 2, 1)
    plt.title('Noise Image')
    plt.imshow(noise_image_rgb)
    plt.axis('off')

    # Gaussian Blur
    plt.subplot(2, 2, 2)
    plt.title('Gaussian Blurred Image')
    plt.imshow(gaussian_denoised_img_rgb)
    plt.axis('off')

    # Median Blur
    plt.subplot(2, 2, 3)
    plt.title('Median Blurred Image')
    plt.imshow(median_denoised_img_rgb)
    plt.axis('off')

    # Bilateral Filter
    plt.subplot(2, 2, 4)
    plt.title('Bilateral Denoised Image')
    plt.imshow(bilateral_denoised_img_rgb)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example file path
denoise_image('images.jpg')
