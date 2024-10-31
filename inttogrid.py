from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import os


import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from urllib.request import urlretrieve


def get_tff(zip_url, extract_to="/usr/share/fonts/truetype/"):
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
        root = Path(extract_to)
        zip_file_path = f"{root}.zip"  # Local name for the zip file
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
    





class DigitRenderer:
    def __init__(self, font_size=128, font_path=None, debug=True):
        """
        Initialize the digit renderer with a specific font size.
        """
        self.font_path = font_path
        self.font_size = font_size
        self.debug = debug
        self.font = self._load_font()
        self.using_fallback = isinstance(self.font, ImageFont.ImageFont)
        
    
    def _load_font(self):
        if self.font_path:
            return ImageFont.truetype(self.font_path, size=self.font_size)
        else:
          return self._load_liberation_font()

    def _load_liberation_font(self):

        """Load Liberation Sans font or fall back to custom renderer"""
        liberation_paths = [
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
            '/usr/share/fonts/truetype/liberation/Liberation_Sans-Regular.ttf',
            '/usr/share/fonts/liberation/LiberationSans-Regular.ttf'
        ]
        
        for path in liberation_paths:
            if os.path.exists(path):
                try:
                    if self.debug:
                        print(f"Loading font: {path}")
                    return ImageFont.truetype(path, size=self.font_size)
                except Exception as e:
                    if self.debug:
                        print(f"Failed to load {path}: {e}")
        
        return ImageFont.load_default()  # This will trigger our fallback renderer

    def _draw_segment(self, draw, x, y, segment_type, width, height, thickness):
        """Draw a single segment of the 7-segment display"""
        if segment_type == 'horizontal':
            points = [
                (x, y + thickness//2),
                (x + thickness//2, y),
                (x + width - thickness//2, y),
                (x + width, y + thickness//2),
                (x + width - thickness//2, y + thickness),
                (x + thickness//2, y + thickness)
            ]
        else:  # vertical
            points = [
                (x + thickness//2, y),
                (x + thickness, y + thickness//2),
                (x + thickness, y + height - thickness//2),
                (x + thickness//2, y + height),
                (x, y + height - thickness//2),
                (x, y + thickness//2)
            ]
        draw.polygon(points, fill=255)

    def _draw_seven_segment(self, digit, img_width, img_height):
        """Draw a digit using 7-segment display style"""
        # Define segment patterns for each digit
        patterns = {
            0: 'abcdef',
            1: 'bc',
            2: 'abged',
            3: 'abgcd',
            4: 'fgbc',
            5: 'afgcd',
            6: 'afgecd',
            7: 'abc',
            8: 'abcdefg',
            9: 'afgbcd'
        }
        
        # Create new image
        img = Image.new('L', (img_width, img_height), color=0)
        draw = ImageDraw.Draw(img)
        
        # Calculate dimensions
        padding = img_width // 10
        inner_width = img_width - 2 * padding
        inner_height = img_height - 2 * padding
        thickness = max(inner_height // 8, 2)
        segment_width = inner_width - thickness
        segment_height = (inner_height - thickness) // 2
        
        # Segment positions relative to top-left corner
        segments = {
            'a': ('horizontal', padding + thickness//2, padding),
            'b': ('vertical', padding + segment_width, padding + thickness//2),
            'c': ('vertical', padding + segment_width, padding + segment_height + thickness),
            'd': ('horizontal', padding + thickness//2, padding + 2 * segment_height + thickness),
            'e': ('vertical', padding, padding + segment_height + thickness),
            'f': ('vertical', padding, padding + thickness//2),
            'g': ('horizontal', padding + thickness//2, padding + segment_height + thickness//2)
        }
        
        # Draw active segments based on the digit
        pattern = patterns.get(digit, '')
        for segment in pattern:
            segment_type, x, y = segments[segment]
            self._draw_segment(draw, x, y, segment_type, segment_width, segment_height, thickness)
            
        return img

    def digit_to_grid(self, digit, size):
        """
        Convert a digit (0-99) to a square binary grid.
        
        Args:
            digit (int): Number between 0-99
            size (int): Size of the square grid (both width and height)
            
        Returns:
            numpy.ndarray: Binary grid representing the digit(s)
        """
        if not 0 <= digit <= 99:
            raise ValueError("Digit must be between 0-99")
        
        scale_factor = 4
        buffer_size = self.font_size // 4
        
        # Calculate dimensions
        img_width = self.font_size * (2 if digit > 9 else 1) + buffer_size * 2
        img_height = self.font_size + buffer_size * 2
        
        if self.using_fallback:
            # Use custom 7-segment display for each digit
            if digit < 10:
                img = self._draw_seven_segment(digit, img_width * scale_factor, img_height * scale_factor)
            else:
                # Create image for double digits
                img = Image.new('L', (img_width * scale_factor, img_height * scale_factor), color=0)
                # Draw each digit separately
                tens = self._draw_seven_segment(digit // 10, img_width * scale_factor // 2, img_height * scale_factor)
                ones = self._draw_seven_segment(digit % 10, img_width * scale_factor // 2, img_height * scale_factor)
                # Paste them side by side
                img.paste(tens, (0, 0))
                img.paste(ones, (img_width * scale_factor // 2, 0))
        else:
            # Use the loaded font
            img = Image.new('L', (img_width * scale_factor, img_height * scale_factor), color=0)
            draw = ImageDraw.Draw(img)
            text = str(digit)
            
            try:
                text_bbox = draw.textbbox((0, 0), text, font=self.font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except AttributeError:
                text_width = len(text) * self.font_size * scale_factor
                text_height = self.font_size * scale_factor
            
            x = (img_width * scale_factor - text_width) // 2
            y = (img_height * scale_factor - text_height) // 2
            
            for dx, dy in [(0,0), (1,0), (0,1)]:
                draw.text((x + dx, y + dy), text, font=self.font, fill=255)
        
        # Apply slight blur to reduce aliasing
        img = img.filter(ImageFilter.GaussianBlur(radius=scale_factor/3))
        
        # Convert to numpy array and get the non-empty bounding box
        digit_array = np.array(img)
        rows = np.any(digit_array > 10, axis=1)
        cols = np.any(digit_array > 10, axis=0)
        trim_array = digit_array[rows][:, cols]
        
        # Convert back to image for high-quality resizing
        trim_img = Image.fromarray(trim_array)
        
        # Calculate padding to make it square
        max_dim = max(trim_img.size)
        square_img = Image.new('L', (max_dim, max_dim), color=0)
        paste_x = (max_dim - trim_img.size[0]) // 2
        paste_y = (max_dim - trim_img.size[1]) // 2
        square_img.paste(trim_img, (paste_x, paste_y))
        
        # Resize to target size using high-quality downsampling
        resized_img = square_img.resize((size, size), Image.LANCZOS)
        
        # Convert to binary with improved thresholding
        final_array = np.array(resized_img)
        threshold = (final_array.max() - final_array.min()) / 2
        return (final_array > threshold).astype(int)

def display_grid(grid):
    """Display a binary grid using Unicode block characters"""
    for row in grid:
        print(''.join(['█' if cell else ' ' for cell in row]))
        



# Example usage
if __name__ == "__main__":
    renderer = DigitRenderer(debug=True)
    size = 20
    
    for number in [7, 42, 99]:
        print(f"\nNumber {number} in {size}x{size} grid:")
        grid = renderer.digit_to_grid(number, size)
        display_grid(grid)