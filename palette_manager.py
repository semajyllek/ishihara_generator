from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import random
import yaml




# Use os.path.join to create the full path to palettes.yaml
PALETTE_PATH = Path(__file__) /  'palettes.yaml'


@dataclass
class ColorSet:
    background: List[str]
    figure: List[str]
    border: str
    background_base: str

@dataclass
class Metadata:
    author: str
    created: str
    contrast_ratio: float
    tags: List[str]

@dataclass
class Palette:
    name: str
    description: str
    type: str
    colors: ColorSet
    metadata: Metadata

class PaletteManager:
    def __init__(self, palette_file: str = PALETTE_PATH):
        self.palette_file = Path(palette_file)
        self.palettes: Dict[str, Palette] = {}
        self.load_palettes()

    def load_palettes(self):
        """Load palettes from YAML file"""
        with open(self.palette_file, 'r') as f:
            data = yaml.safe_load(f)
            
        for key, palette_data in data['deuteranopia_palettes'].items():
            colors = ColorSet(**palette_data['colors'])
            metadata = Metadata(**palette_data['metadata'])
            
            self.palettes[key] = Palette(
                name=palette_data['name'],
                description=palette_data['description'],
                type=palette_data['type'],
                colors=colors,
                metadata=metadata
            )

    def get_random_palette(self) -> Palette:
        """Get a random palette"""
        return random.choice(list(self.palettes.values()))

    def get_palette_by_name(self, name: str) -> Optional[Palette]:
        """Get a specific palette by name"""
        return self.palettes.get(name)
    

