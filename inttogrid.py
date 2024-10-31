





import os
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from urllib.request import urlretrieve

FONT_SIZE = 50
SIZE = 50


# prompt: create a function, get_damion_tff  which will download and unzip the given tff zip location on the internet
# !wget https://www.1001fonts.com/download/damion.zip
# !unzip damion.zip  # assume Damion-Regular.tff is in the path

def get_damion_tff(zip_url, extract_to="/usr/share/fonts/truetype/"):
    """
    Downloads and extracts a TTF font file from a given zip URL.

    Args:
        zip_url: The URL of the zip file containing the TTF font.
        extract_to: The directory to extract the font file to.
                     Defaults to /usr/share/fonts/truetype/.
    Returns:
        The path to the extracted TTF file if successful, None otherwise.
    """
    
    try:
        # Create the target directory if it doesn't exist
        os.makedirs(extract_to, exist_ok=True)

        # Download the zip file
        zip_file_path = "damion.zip"  # Local name for the zip file
        urlretrieve(zip_url, zip_file_path)

        # Extract the zip file
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            for member in zip_ref.namelist():
                if member.lower().endswith(".ttf") or member.lower().endswith(".otf"):  # Check for TTF or OTF
                    zip_ref.extract(member, extract_to)
                    ttf_file_path = os.path.join(extract_to, member)
                    return ttf_file_path
        
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def number_to_image(number, font_path, font_size, image_size=(SIZE, SIZE)):
    """
    Converts a number to an image using the specified font.
    """
    # Create a new image
    image = Image.new('1', image_size, "white")
    draw = ImageDraw.Draw(image)
    # Load the font
    font = ImageFont.truetype(font_path, font_size)
    # Get text size
    bbox = draw.textbbox((0, 0), str(number), font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Calculate x and y coordinates to center the text
    x = (image_size[0] - text_width) / 2
    y = (image_size[1] - text_height) / 2 - 20 # Move the number up by 10 pixels
    # Draw the text
    draw.text((x, y), str(number), font=font, fill="black")
    return image


def image_to_binary_grid(image):
  """Converts a binary image to a binary grid.

  Args:
    image: A PIL Image object representing a binary image.

  Returns:
    A list of lists representing the binary grid.
  """
  # Ensure the image is in '1' mode (binary)
  if image.mode != '1':
    image = image.convert('1')
    
  # Get image dimensions
  width, height = image.size

  # Create the binary grid
  binary_grid = []
  for y in range(height):
    row = []
    for x in range(width):

      # Get pixel value (0 for black, 255 for white)
      pixel_value = image.getpixel((x, y))

      # Convert to binary (0 or 1)
      row.append(0 if pixel_value == 255 else 1)
    binary_grid.append(row)
    
  return binary_grid



def generate_binary_image_with_font(num, font_path, font_size=FONT_SIZE, size=SIZE):
    """
    Generates image for given number using the preset font path.
    """
    img = number_to_image(num, font_path, font_size, image_size=(size, size))
    img_grid = image_to_binary_grid(img)
    return img_grid, img
           