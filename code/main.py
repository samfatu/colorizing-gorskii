import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import sys

def get_all_images_in_directory(path):
  files = [f for f in listdir(path) if isfile(join(path, f))]
  return [f for f in files if f.split('.')[1] == 'jpg']


def read_image(filename):
  image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
  return image


def get_image_channels(image):
  img_height, img_width = image.shape
  channel_height = img_height // 3

  image_with_channels = np.zeros((channel_height, img_width, 3), dtype='uint8')

  image_with_channels[:,:,0] = image[0:channel_height, :]   # Blue channel
  image_with_channels[:,:,1] = image[channel_height:channel_height * 2, :]   # Green channel
  image_with_channels[:,:,2] = image[channel_height * 2:channel_height * 3, :]   # Red channel

  # Image with all channels set and with color palette BGR
  return image_with_channels


def split_image_to_channels(image):
  return image[:,:,0], image[:,:,1], image[:,:,2]


def crop_borders(image, rate=0.05):
  height = image.shape[0]
  width = image.shape[1]

  height_crop_pixel = int(height * rate)
  width_crop_pixel = int(width * rate)

  # Crop the image with start and end pixel values
  return image[height_crop_pixel : height - height_crop_pixel, width_crop_pixel : width - width_crop_pixel, :]

# SSD calculation
def calculate_difference(channel1, shifted_channel2):
  return np.sum(np.square(channel1 - shifted_channel2))

def calculate_ssd(channel1, channel2, isLaplacian):
  window_size = 15  # [-15, 15] window size
  coordinates = [] # Shift coordinates
  min_distance = float('inf')   # Start with a huge number

  if isLaplacian:
    #Apply Laplacian Filtering to calculate SSD with looking at edges
    channel1 = cv2.Laplacian(channel1, cv2.CV_64F)
    channel2 = cv2.Laplacian(channel2, cv2.CV_64F)

  for i in np.arange(-window_size, window_size):
    for j in np.arange(-window_size, window_size):
      shift_channel2 = np.roll(channel2, [i, j], axis=(0, 1))
      distance = calculate_difference(channel1, shift_channel2)

      # If calculated differance is less than minimum distance value so far, set it and set the shift coordinates
      if distance <= min_distance:
        min_distance = distance
        coordinates = [i, j]

  return coordinates

def match_channels(image, isLaplacian):
  b, g, r = split_image_to_channels(image)

  R2B_coordinates = calculate_ssd(b, r, isLaplacian)
  G2B_coordinates = calculate_ssd(b, g, isLaplacian)

  print(f'Red channel shifted by: {R2B_coordinates}\tGreen channel shifted by: {G2B_coordinates}')
  # Apply calculated shift values
  alignedG = np.roll(g, G2B_coordinates, axis=(0, 1))
  alignedR = np.roll(r, R2B_coordinates, axis=(0, 1))

  # Return them as RGB color space
  return (np.dstack([alignedR, alignedG ,b]), G2B_coordinates, R2B_coordinates)

def gamma_correction(image):
  i_gamma = 1 / 0.7
  lut = np.array([((i / 255) ** i_gamma) * 255 for i in np.arange(0, 256)], dtype = 'uint8')
  return cv2.LUT(image, lut)

def sharpen_image(image):
  kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
  return cv2.filter2D(image, -1, kernel)

def equalize_histogram(image):
  yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
  yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0]) # Y Channel
  return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

def run(isLaplacian = True, gamma = False, sharpen = False, eqHistogram = False):
  all_image_paths = get_all_images_in_directory('../html/images/raw_data/')

  crop_rate = 0.05

  for image_path in all_image_paths:
    print(f'Image: {image_path}')
    outputFileName = image_path.split(".")[0] + "_L" if isLaplacian else image_path.split(".")[0]
    image = read_image(f'../html/images/raw_data/{image_path}')
    colored_image = get_image_channels(image)
    cropped_image = crop_borders(colored_image, crop_rate)
    RGBmatched, G_shift_coord, R_shift_coord = match_channels(cropped_image, isLaplacian)
    outputImage = RGBmatched

    if gamma:
      outputImage = gamma_correction(outputImage)
      outputFileName += "_G"

    if sharpen:
      outputImage = sharpen_image(outputImage)
      outputFileName += "_S"

    if eqHistogram:
      outputImage = equalize_histogram(outputImage)
      outputFileName += "_H"


    figures = [cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB), RGBmatched, outputImage]
    titles = ["Colorized version before alignment", "Algined version", "Filtered version after alignment"]
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(image_path)
    ax = []
    for i in range(3):
      ax.append(fig.add_subplot(1, 3, i+1))
      ax[-1].set_title(titles[i])
      if i == 1:
        ax[-1].set_xlabel(f'G: {G_shift_coord}, R: {R_shift_coord}')
      ax[-1].set_xticks([])
      ax[-1].set_yticks([])
      plt.imshow(figures[i])
      # To save plot
      #plt.savefig(f'{outputFileName}.jpg', bbox_inches='tight')
    plt.show()

    cv2.imwrite(f'../output/{outputFileName}.jpg', cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
  laplacian, gamma, sharpen, hist = False, False, False, False

  for arg in sys.argv:
    if arg == "-l":
      laplacian = True
    elif arg == "-g":
      gamma = True
    elif arg == "-s":
      sharpen = True
    elif arg == "-h":
      hist = True

  run(isLaplacian=laplacian, gamma=gamma, sharpen=sharpen, eqHistogram=hist)