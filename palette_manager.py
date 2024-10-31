# palette_manager.py
from color_generator import generate_ishihara_palette

class PaletteManager:
    def __init__(self):
        self.current_palette = None
        self.generate_new_palette()
    
    def generate_new_palette(self):
        """Generate a fresh palette for each plate"""
        palette_data = generate_ishihara_palette()
        self.current_palette = palette_data['colors']
        return self.current_palette
    
    def get_random_palette(self):
        """Generate a new random palette each time"""
        return self.generate_new_palette()