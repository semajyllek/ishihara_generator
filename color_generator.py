# color_generator.py
from colorspacious import cspace_converter, CVD_SPACE
from colorharmonies import Color, MonochromaticColor
import numpy as np

class ColorPaletteGenerator:
    def __init__(self):
        self.cvd_space = CVD_SPACE
        self.lab_to_rgb = cspace_converter("CAM02-UCS", "sRGB1")
        self.rgb_to_lab = cspace_converter("sRGB1", "CAM02-UCS")

    def generate_background_palette(self, base_hue, num_colors=10):
        colors = []
        base_color = Color(HSL=(base_hue, 0.8, 0.6))
        variations = MonochromaticColor(base_color, num_colors)
        
        for color in variations:
            rgb = color.rgb
            lab = self.rgb_to_lab(rgb)
            lab[0] += np.random.uniform(-10, 10)
            lab[1] *= 1.2
            rgb_adjusted = self.lab_to_rgb(lab)
            rgb_adjusted = np.clip(rgb_adjusted, 0, 1)
            colors.append(rgb_adjusted)
            
        return [self.rgb_to_hex(color) for color in colors]

    def generate_figure_palette(self, background_hue, num_colors=10):
        figure_hue = (background_hue + 180) % 360
        
        colors = []
        base_color = Color(HSL=(figure_hue, 0.9, 0.5))
        variations = MonochromaticColor(base_color, num_colors)
        
        for color in variations:
            rgb = color.rgb
            lab = self.rgb_to_lab(rgb)
            lab[0] += np.random.uniform(-5, 5)
            lab[1] *= 1.3
            rgb_adjusted = self.lab_to_rgb(lab)
            rgb_adjusted = np.clip(rgb_adjusted, 0, 1)
            colors.append(rgb_adjusted)
            
        return [self.rgb_to_hex(color) for color in colors]

    @staticmethod
    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )

def generate_ishihara_palette():
    color_gen = ColorPaletteGenerator()
    
    # Generate different color themes
    themes = [
        {'hue': 30, 'name': 'warm_orange'},  # Orange
        {'hue': 0, 'name': 'warm_red'},      # Red
        {'hue': 280, 'name': 'cool_purple'}, # Purple
        {'hue': 45, 'name': 'warm_yellow'},  # Yellow
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