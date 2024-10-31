# color_generator.py
from colorharmonies import Color, MonochromaticColor, ComplementaryColor
import numpy as np

class ColorPaletteGenerator:
    def generate_background_palette(self, base_hue, num_colors=10):
        colors = []
        # Use higher saturation and lightness for background
        base_color = Color(HSL=(base_hue, 0.7, 0.65))
        
        # Generate monochromatic variations
        variations = MonochromaticColor(base_color, num_colors)
        
        for color in variations:
            colors.append(self.rgb_to_hex(color.rgb))
        return colors

    def generate_figure_palette(self, background_hue, num_colors=10):
        # Use complementary color for maximum contrast
        figure_hue = (background_hue + 180) % 360
        
        colors = []
        # Use higher saturation and lower lightness for figure
        base_color = Color(HSL=(figure_hue, 0.8, 0.45))
        variations = MonochromaticColor(base_color, num_colors)
        
        for color in variations:
            colors.append(self.rgb_to_hex(color.rgb))
        return colors

    @staticmethod
    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )

def generate_ishihara_palette():
    color_gen = ColorPaletteGenerator()
    
    # Define themes optimized for deuteranopia testing
    themes = [
        {'hue': 30, 'name': 'warm_orange'},    # Orange background with blue-green figures
        {'hue': 15, 'name': 'warm_coral'},     # Coral background with blue figures
        {'hue': 45, 'name': 'warm_yellow'},    # Yellow background with blue-purple figures
        {'hue': 200, 'name': 'cool_blue'},     # Blue background with orange figures
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