# color_generator.py
from colorharmonies import Color
import numpy as np

class ColorPaletteGenerator:
    def generate_background_palette(self, base_hue, num_colors=10):
        colors = []
        # Create base color with high saturation and medium lightness
        base_color = Color(HLS=(base_hue/360, 0.7, 0.65))  # Note: hue needs to be 0-1
        colors.append(base_color)
        
        # Generate variations spreading around the base hue
        spread = 30  # degrees
        step = spread / (num_colors - 1) if num_colors > 1 else 0
        
        for i in range(num_colors - 1):
            # Alternate between plus and minus from base hue
            if i % 2 == 0:
                hue_variation = ((base_hue + step * (i//2)) % 360) / 360  # Convert to 0-1
            else:
                hue_variation = ((base_hue - step * (i//2)) % 360) / 360  # Convert to 0-1
                
            # Vary saturation and lightness slightly
            saturation = 0.7 + np.random.uniform(-0.1, 0.1)
            lightness = 0.65 + np.random.uniform(-0.1, 0.1)
            
            # Keep values in valid ranges
            saturation = np.clip(saturation, 0.5, 0.9)
            lightness = np.clip(lightness, 0.5, 0.8)
            
            new_color = Color(HLS=(hue_variation, saturation, lightness))
            colors.append(new_color)
            
        return [self.rgb_to_hex(color.rgb) for color in colors]

    def generate_figure_palette(self, background_hue, num_colors=10):
        colors = []
        # Use complementary color for maximum contrast
        figure_hue = (background_hue + 180) % 360
        
        # Create base complementary color
        base_color = Color(HLS=(figure_hue/360, 0.8, 0.45))
        colors.append(base_color)
        
        # Generate variations
        spread = 30  # degrees
        step = spread / (num_colors - 1) if num_colors > 1 else 0
        
        for i in range(num_colors - 1):
            if i % 2 == 0:
                hue_variation = ((figure_hue + step * (i//2)) % 360) / 360
            else:
                hue_variation = ((figure_hue - step * (i//2)) % 360) / 360
                
            saturation = 0.8 + np.random.uniform(-0.1, 0.1)
            lightness = 0.45 + np.random.uniform(-0.1, 0.1)
            
            # Keep values in valid ranges
            saturation = np.clip(saturation, 0.6, 0.9)
            lightness = np.clip(lightness, 0.3, 0.6)
            
            new_color = Color(HLS=(hue_variation, saturation, lightness))
            colors.append(new_color)
            
        return [self.rgb_to_hex(color.rgb) for color in colors]

    @staticmethod
    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )

def generate_ishihara_palette():
    color_gen = ColorPaletteGenerator()
    
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