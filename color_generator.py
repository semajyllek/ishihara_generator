# color_generator.py
from colorharmonies import Color, Harmonies
import numpy as np

class ColorPaletteGenerator:
    def generate_background_palette(self, base_hue, num_colors=10):
        colors = []
        # Create base color with high saturation and medium lightness
        base_color = Color(HSL=(base_hue, 0.7, 0.65))
        
        # Use harmonies to generate variations
        analogous_colors = Harmonies.analogous(base_color)
        
        # Start with base and analogous colors
        colors = [base_color] + analogous_colors
        
        # Generate additional variations using the base color as reference
        while len(colors) < num_colors:
            # Vary the base color properties slightly
            hue_variation = base_hue + np.random.uniform(-15, 15)
            saturation = base_color.hsl[1] + np.random.uniform(-0.1, 0.1)
            lightness = base_color.hsl[2] + np.random.uniform(-0.1, 0.1)
            
            # Keep values in valid ranges
            saturation = np.clip(saturation, 0.5, 0.9)
            lightness = np.clip(lightness, 0.5, 0.8)
            
            new_color = Color(HSL=(hue_variation, saturation, lightness))
            colors.append(new_color)
            
        return [self.rgb_to_hex(color.rgb) for color in colors[:num_colors]]

    def generate_figure_palette(self, background_hue, num_colors=10):
        # Use complementary color for maximum contrast
        figure_hue = (background_hue + 180) % 360
        
        # Create base complementary color
        base_color = Color(HSL=(figure_hue, 0.8, 0.45))
        
        # Use harmonies to generate variations
        triad_colors = Harmonies.triad(base_color)
        colors = [base_color] + triad_colors
        
        # Generate additional variations using the base color as reference
        while len(colors) < num_colors:
            hue_variation = figure_hue + np.random.uniform(-15, 15)
            saturation = base_color.hsl[1] + np.random.uniform(-0.1, 0.1)
            lightness = base_color.hsl[2] + np.random.uniform(-0.1, 0.1)
            
            # Keep values in valid ranges
            saturation = np.clip(saturation, 0.6, 0.9)
            lightness = np.clip(lightness, 0.3, 0.6)
            
            new_color = Color(HSL=(hue_variation, saturation, lightness))
            colors.append(new_color)
            
        return [self.rgb_to_hex(color.rgb) for color in colors[:num_colors]]

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