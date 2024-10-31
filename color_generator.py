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


    def generate_background_palette(self, base_hue, condition, num_colors=12):
        colors = []
        
        if condition == 'deuteranopia':
            # Create color groups relative to the base hue
            color_groups = [
                # Main base color group
                {'hue': base_hue, 'sat': (0.3, 0.5), 'light': (0.4, 0.6)},
                # Slightly warmer variation
                {'hue': (base_hue - 20) % 360, 'sat': (0.2, 0.4), 'light': (0.5, 0.7)},
                # Slightly cooler variation
                {'hue': (base_hue + 20) % 360, 'sat': (0.2, 0.3), 'light': (0.5, 0.6)}
            ]
            
            # Generate colors from each group
            for group in color_groups:
                for _ in range(num_colors // 3):
                    h = (group['hue'] + np.random.uniform(-10, 10)) / 360
                    s = np.random.uniform(*group['sat'])
                    l = np.random.uniform(*group['light'])
                    colors.append(self.create_color(h, s, l))

        # Fill remaining slots with variations
        while len(colors) < num_colors:
            group = np.random.choice(color_groups)
            h = (group['hue'] + np.random.uniform(-15, 15)) / 360
            s = np.random.uniform(*group['sat'])
            l = np.random.uniform(*group['light'])
            colors.append(self.create_color(h, s, l))

        return [self.rgb_to_hex(color.RGB) for color in colors]

def generate_ishihara_palette():
    color_gen = ColorPaletteGenerator()
    
    # Traditional Ishihara themes with proper base hues
    themes = [
        {
            'hue': 80,  # Sage green base
            'name': 'traditional_sage',
            'condition': 'deuteranopia',
            'figure_hue': 20,  # Coral
            'border': '#E8D0A9',
            'background_base': '#F5F2E8'
        },
        {
            'hue': 50,  # Earth tone base
            'name': 'traditional_earth',
            'condition': 'deuteranopia',
            'figure_hue': 20,  # Coral
            'border': '#E8D0A9',
            'background_base': '#F5F2E8'
        },
        {
            'hue': 95,  # Olive base
            'name': 'traditional_olive',
            'condition': 'deuteranopia',
            'figure_hue': 20,  # Coral
            'border': '#E8D0A9',
            'background_base': '#F5F2E8'
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

if __name__ == "__main__":
    palette = generate_ishihara_palette()
    print(palette['name'])
    print('Background:', palette['colors']['background'])
    print('Figure:', palette['colors']['figure'])
    print('Border:', palette['colors']['border'])
    print('Background Base:', palette['colors']['background_base'])