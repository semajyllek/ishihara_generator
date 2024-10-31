# color_generator.py
from colorharmonies import Color
import numpy as np
import colorsys

class ColorPaletteGenerator:
    @staticmethod
    def rgb_to_hex(RGB):
        """Convert RGB values (0-1) to hex color string"""
        return '#{:02x}{:02x}{:02x}'.format(
            int(RGB[0] * 255),
            int(RGB[1] * 255),
            int(RGB[2] * 255)
        )
    def create_color(self, h, s, l):
        """Create a Color object with all required color spaces"""
        RGB = colorsys.hls_to_rgb(h, l, s)
        HSV = colorsys.rgb_to_hsv(*RGB)
        return Color(RGB=RGB, HLS=(h, l, s), HSV=HSV)

    def generate_background_palette(self, base_hue, condition, num_colors=15):
        colors = []
        # Adjust parameters based on type of color blindness test
        if condition == 'deuteranopia':
            # Salmon-orange range optimized for deuteranopia testing
            saturations = np.linspace(0.75, 0.85, 5)
            lightnesses = np.linspace(0.60, 0.75, 3)
            hue_variation = 10
        elif condition == 'protanopia':
            # Red range optimized for protanopia testing
            saturations = np.linspace(0.70, 0.85, 5)
            lightnesses = np.linspace(0.55, 0.70, 3)
            hue_variation = 8
        else:  # tritanopia
            # Brown range optimized for tritanopia testing
            saturations = np.linspace(0.65, 0.80, 5)
            lightnesses = np.linspace(0.50, 0.65, 3)
            hue_variation = 12

        # Generate base colors with controlled variations
        for s in saturations:
            for l in lightnesses:
                h = ((base_hue + np.random.uniform(-hue_variation, hue_variation)) % 360) / 360
                colors.append(self.create_color(h, s, l))

        # Add some subtle variations
        while len(colors) < num_colors:
            h = ((base_hue + np.random.uniform(-hue_variation, hue_variation)) % 360) / 360
            s = np.random.choice(saturations) + np.random.uniform(-0.05, 0.05)
            l = np.random.choice(lightnesses) + np.random.uniform(-0.05, 0.05)
            s = np.clip(s, 0.6, 0.9)
            l = np.clip(l, 0.5, 0.8)
            colors.append(self.create_color(h, s, l))

        return [self.rgb_to_hex(color.RGB) for color in colors]

    def generate_figure_palette(self, base_hue, condition, num_colors=15):
        colors = []
        # Adjust parameters based on type of color blindness test
        if condition == 'deuteranopia':
            # Green range optimized for deuteranopia testing
            saturations = np.linspace(0.75, 0.90, 5)
            lightnesses = np.linspace(0.40, 0.55, 3)
            hue_variation = 8
        elif condition == 'protanopia':
            # Blue-green range optimized for protanopia testing
            saturations = np.linspace(0.70, 0.85, 5)
            lightnesses = np.linspace(0.45, 0.60, 3)
            hue_variation = 10
        else:  # tritanopia
            # Blue range optimized for tritanopia testing
            saturations = np.linspace(0.70, 0.85, 5)
            lightnesses = np.linspace(0.45, 0.60, 3)
            hue_variation = 12

        # Generate base colors with controlled variations
        for s in saturations:
            for l in lightnesses:
                h = ((base_hue + np.random.uniform(-hue_variation, hue_variation)) % 360) / 360
                colors.append(self.create_color(h, s, l))

        # Add some subtle variations
        while len(colors) < num_colors:
            h = ((base_hue + np.random.uniform(-hue_variation, hue_variation)) % 360) / 360
            s = np.random.choice(saturations) + np.random.uniform(-0.05, 0.05)
            l = np.random.choice(lightnesses) + np.random.uniform(-0.05, 0.05)
            s = np.clip(s, 0.65, 0.95)
            l = np.clip(l, 0.35, 0.65)
            colors.append(self.create_color(h, s, l))

        return [self.rgb_to_hex(color.RGB) for color in colors]

def generate_ishihara_palette():
    color_gen = ColorPaletteGenerator()
    
    # Themes specifically calibrated for color blindness testing
    themes = [
        {
            'hue': 25,  # Salmon-orange
            'name': 'deuteranopia_test_1',
            'condition': 'deuteranopia',
            'figure_hue': 120,  # Green
            'border': '#E8D0A9',
            'background_base': '#FFF6E9'
        },
        {
            'hue': 15,  # Warmer orange
            'name': 'deuteranopia_test_2',
            'condition': 'deuteranopia',
            'figure_hue': 125,  # Slightly different green
            'border': '#E8D0A9',
            'background_base': '#FFF6E9'
        },
        {
            'hue': 30,  # Orange
            'name': 'protanopia_test',
            'condition': 'protanopia',
            'figure_hue': 140,  # Blue-green
            'border': '#E8D0A9',
            'background_base': '#FFF6E9'
        },
        {
            'hue': 20,  # Reddish-brown
            'name': 'tritanopia_test',
            'condition': 'tritanopia',
            'figure_hue': 180,  # Cyan-blue
            'border': '#E8D0A9',
            'background_base': '#FFF6E9'
        }
    ]
    
    theme = np.random.choice(themes)
    background_colors = color_gen.generate_background_palette(
        theme['hue'], 
        theme['condition']
    )
    figure_colors = color_gen.generate_figure_palette(
        theme['figure_hue'], 
        theme['condition']
    )
    
    return {
        'name': theme['name'],
        'colors': {
            'background': background_colors,
            'figure': figure_colors,
            'border': theme['border'],
            'background_base': theme['background_base']
        }
    }
