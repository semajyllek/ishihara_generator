# color_generator.py
from colorharmonies import Color
import numpy as np
import colorsys

class ColorPaletteGenerator:
    def generate_background_palette(self, base_hue, num_colors=15):
        colors = []
        base_hues = [
            (base_hue % 360) / 360,
            ((base_hue + 10) % 360) / 360,
            ((base_hue - 10) % 360) / 360,
        ]
        
        # More varied saturations and lightnesses
        for base_h in base_hues:
            # Vivid variants
            colors.append(self.create_color(base_h, 0.90, 0.65))
            colors.append(self.create_color(base_h, 0.85, 0.70))
            # Rich variants
            colors.append(self.create_color(base_h, 0.80, 0.55))
            colors.append(self.create_color(base_h, 0.75, 0.60))
            # Deep variants
            colors.append(self.create_color(base_h, 0.70, 0.45))
            
        while len(colors) < num_colors:
            h = ((base_hue + np.random.uniform(-15, 15)) % 360) / 360
            s = np.random.uniform(0.70, 0.90)
            l = np.random.uniform(0.45, 0.70)
            colors.append(self.create_color(h, s, l))
            
        return [self.rgb_to_hex(color.RGB) for color in colors]

def generate_ishihara_palette():
    color_gen = ColorPaletteGenerator()
    
    # Rich, natural themes inspired by your suggestions
    themes = [
        {
            'hue': 280,  # Deep purple
            'name': 'blackberry',
        },
        {
            'hue': 45,   # Sandy gold
            'name': 'harvest_sand',
        },
        {
            'hue': 25,   # Rich squash
            'name': 'september_squash',
        },
        {
            'hue': 210,  # Deep ocean blue
            'name': 'deep_ocean',
        },
        {
            'hue': 140,  # Forest green
            'name': 'forest_depths',
        },
        {
            'hue': 15,   # Terra cotta
            'name': 'terra_cotta',
        },
        {
            'hue': 330,  # Berry pink
            'name': 'wild_berry',
        },
        {
            'hue': 35,   # Amber brown
            'name': 'amber_woods',
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