# color_generator.py
from colorharmonies import Color
import numpy as np
import colorsys

class ColorPaletteGenerator:
    def create_color(self, h, s, l):
        """Create a Color object with all required color spaces"""
        RGB = colorsys.hls_to_rgb(h, l, s)
        HSV = colorsys.rgb_to_hsv(*RGB)
        return Color(RGB=RGB, HLS=(h, l, s), HSV=HSV)

    def generate_background_palette(self, base_hue, num_colors=15):  # Increased number of colors
        colors = []
        # Create multiple base colors with varying warmth
        base_hues = [
            (base_hue % 360) / 360,  # Main hue
            ((base_hue + 15) % 360) / 360,  # Warmer variation
            ((base_hue - 15) % 360) / 360,  # Cooler variation
        ]
        
        # Generate more varied saturations and lightnesses
        for base_h in base_hues:
            # Bright variants
            colors.append(self.create_color(base_h, 0.85, 0.65))
            colors.append(self.create_color(base_h, 0.75, 0.70))
            # Medium variants
            colors.append(self.create_color(base_h, 0.70, 0.60))
            colors.append(self.create_color(base_h, 0.65, 0.55))
            # Subtle variants
            colors.append(self.create_color(base_h, 0.60, 0.75))
            
        # Add some random variations for organic feel
        while len(colors) < num_colors:
            h = ((base_hue + np.random.uniform(-20, 20)) % 360) / 360
            s = np.random.uniform(0.6, 0.85)
            l = np.random.uniform(0.55, 0.75)
            colors.append(self.create_color(h, s, l))
            
        return [self.rgb_to_hex(color.RGB) for color in colors]

    def generate_figure_palette(self, background_hue, num_colors=15):  # Increased number
        colors = []
        figure_hue = (background_hue + 180) % 360  # Complementary base
        
        # Create multiple base hues for richer variation
        base_hues = [
            figure_hue / 360,  # Main complementary
            ((figure_hue + 10) % 360) / 360,  # Slight warm shift
            ((figure_hue - 10) % 360) / 360,  # Slight cool shift
        ]
        
        # Generate deep, rich variants
        for base_h in base_hues:
            # Deep variants
            colors.append(self.create_color(base_h, 0.90, 0.35))
            colors.append(self.create_color(base_h, 0.85, 0.40))
            # Medium variants
            colors.append(self.create_color(base_h, 0.80, 0.45))
            colors.append(self.create_color(base_h, 0.75, 0.50))
            # Lighter variants
            colors.append(self.create_color(base_h, 0.70, 0.55))
            
        # Add random variations for organic feel
        while len(colors) < num_colors:
            h = ((figure_hue + np.random.uniform(-15, 15)) % 360) / 360
            s = np.random.uniform(0.70, 0.90)
            l = np.random.uniform(0.35, 0.55)
            colors.append(self.create_color(h, s, l))
            
        return [self.rgb_to_hex(color.RGB) for color in colors]

    @staticmethod
    def rgb_to_hex(RGB):
        return '#{:02x}{:02x}{:02x}'.format(
            int(RGB[0] * 255),
            int(RGB[1] * 255),
            int(RGB[2] * 255)
        )

def generate_ishihara_palette():
    color_gen = ColorPaletteGenerator()
    
    # Enhanced themes with richer base colors
    themes = [
        {
            'hue': 25,  # Warm orange-coral
            'name': 'warm_coral',
        },
        {
            'hue': 15,  # Rich salmon
            'name': 'bright_salmon',
        },
        {
            'hue': 35,  # Golden orange
            'name': 'golden_orange',
        },
        {
            'hue': 8,   # Deep coral
            'name': 'deep_coral',
        }
    ]
    
    theme = np.random.choice(themes)
    background_colors = color_gen.generate_background_palette(theme['hue'])
    figure_colors = color_gen.generate_figure_palette(theme['hue'])
    
    return {
        'name': theme['name'],
        'colors': {
            'background': background_colors,
            'figure': figure_colors,
            'border': '#E8D0A9',
            'background_base': '#FFF6E9'
        }
    }