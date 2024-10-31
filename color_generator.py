# color_generator.py
import numpy as np
import colorsys

class ColorPaletteGenerator:
    def hsl_to_hex(self, h, s, l):
        """Convert HSL color to hex string"""
        rgb = colorsys.hls_to_rgb(h/360, l, s)
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )

    def generate_variations(self, base_hue, num_colors=10, s_range=(0.6, 0.8), l_range=(0.5, 0.7)):
        """Generate color variations around a base hue"""
        colors = []
        for _ in range(num_colors):
            # Slightly vary the hue
            hue = (base_hue + np.random.uniform(-10, 10)) % 360
            # Vary saturation and lightness within ranges
            sat = np.random.uniform(*s_range)
            light = np.random.uniform(*l_range)
            colors.append(self.hsl_to_hex(hue, sat, light))
        return colors

    def generate_background_palette(self, base_hue, num_colors=10):
        """Generate background colors - generally warmer and lighter"""
        return self.generate_variations(
            base_hue,
            num_colors,
            s_range=(0.6, 0.8),
            l_range=(0.6, 0.75)
        )

    def generate_figure_palette(self, background_hue, num_colors=10):
        """Generate figure colors - generally more saturated and darker"""
        # Use complementary color for maximum contrast
        figure_hue = (background_hue + 180) % 360
        return self.generate_variations(
            figure_hue,
            num_colors,
            s_range=(0.7, 0.9),
            l_range=(0.4, 0.6)
        )

def generate_ishihara_palette():
    color_gen = ColorPaletteGenerator()
    
    # Define themes optimized for deuteranopia testing
    themes = [
        {
            'hue': 30,  # Orange
            'name': 'warm_orange',
            'border': '#E8D0A9',
            'background_base': '#FFF6E9'
        },
        {
            'hue': 15,  # Coral
            'name': 'warm_coral',
            'border': '#E8D0A9',
            'background_base': '#FFF6E9'
        },
        {
            'hue': 45,  # Yellow
            'name': 'warm_yellow',
            'border': '#E8D0A9',
            'background_base': '#FFF6E9'
        },
        {
            'hue': 200,  # Blue
            'name': 'cool_blue',
            'border': '#E8D0A9',
            'background_base': '#FFF6E9'
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
            'border': theme['border'],
            'background_base': theme['background_base']
        }
    }