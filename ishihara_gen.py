
from pathlib import Path


import math
import random
import numpy as np


import pygame
import pymunk
import pymunk.pygame_util

from PIL import Image, ImageDraw


import yaml
from typing import Optional
import random

from .palette_manager import PaletteManager
from .inttogrid import DigitRenderer
from .number_grids import NUMBER_FIVE

# Fixed constants for circle sizes
LARGE_CIRCLE_DIAMETER = 800  # pixels
SMALL_CIRCLE_DIAMETERS = [40, 44, 48, 52]  # pixels
GRID_SIZE = LARGE_CIRCLE_DIAMETER // 20
FONT_SIZE = 128
DEMO_NUMBER = 5


class IshiharaPlateGenerator:
    def __init__(self, palette_manager: Optional[PaletteManager] = None, num: int = DEMO_NUMBER):
        
        self.main_circle_radius = LARGE_CIRCLE_DIAMETER // 2
        self.small_circle_radii = [d // 2 for d in SMALL_CIRCLE_DIAMETERS]
        self.max_small_radius = max(self.small_circle_radii)

        self.rect_width = LARGE_CIRCLE_DIAMETER
        self.rect_height = LARGE_CIRCLE_DIAMETER

        self.width = LARGE_CIRCLE_DIAMETER + 200
        self.height = LARGE_CIRCLE_DIAMETER + 200

        self.space = pymunk.Space()
        self.space.gravity = (0.0, 900.0)

        self.space.collision_slop = 0.0
        self.space.collision_bias = pow(1.0 - 0.3, 60.0)
        self.space.iterations = 30

        self.center_x = self.width // 2
        self.center_y = self.height // 2
        self.palette_manager = palette_manager or PaletteManager()
        selected_palette = self.palette_manager.get_random_palette()
        
        self.background_colors = selected_palette.colors.background
        self.figure_colors = selected_palette.colors.figure
        self.border_color = selected_palette.colors.border
        self.background_base = selected_palette.colors.background_base
        

        self.current_bg_color_index = 0
        self.current_fg_color_index = 0

        self.create_boundary()

        # creates binary grids with stylistic integer masks
        self.renderer = DigitRenderer(font_size=FONT_SIZE, debug=True)
        self.bin_num = self.renderer.digit_to_grid(digit=number, size=(GRID_SIZE, GRID_SIZE))

    def get_next_background_color(self):
        color = self.background_colors[self.current_bg_color_index]
        self.current_bg_color_index = (self.current_bg_color_index + 1) % len(self.background_colors)
        return color

    def get_next_figure_color(self):
        color = self.figure_colors[self.current_fg_color_index]
        self.current_fg_color_index = (self.current_fg_color_index + 1) % len(self.figure_colors)
        return color

    def is_inside_main_circle(self, x, y):
        dx = x - self.center_x
        dy = y - self.center_y
        return dx*dx + dy*dy <= self.main_circle_radius * self.main_circle_radius


    def is_inside_number(self, x, y):
        # Convert coordinates to number grid space
        grid_size = self.bin_num.shape
        number_width = self.main_circle_radius * 1.4   # Made slightly larger
        number_height = self.main_circle_radius * 1.4

        # Center the number with slight upward shift
        number_x = self.center_x - number_width/2
        number_y = self.center_y - number_height/2 - self.main_circle_radius * 0.1

        # Convert point to grid coordinates
        grid_x = int((x - number_x) / (number_width / grid_size[1]))
        grid_y = int((y - number_y) / (number_height / grid_size[0]))

        # Check if point is inside grid and is part of number
        if 0 <= grid_x < grid_size[1] and 0 <= grid_y < grid_size[0]:
            return NUMBER_FIVE[grid_y, grid_x] == 1
        return False

    def create_boundary(self):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.space.add(body)

        rect_left = self.center_x - self.rect_width//2
        rect_right = self.center_x + self.rect_width//2
        rect_top = self.center_y - self.rect_height//2
        rect_bottom = self.center_y + self.rect_height//2

        segments = [
            [(rect_left, rect_bottom), (rect_right, rect_bottom)],
            [(rect_left, rect_bottom), (rect_left, rect_top)],
            [(rect_right, rect_bottom), (rect_right, rect_top)]
        ]

        for points in segments:
            segment = pymunk.Segment(body, points[0], points[1], 1.0)
            segment.friction = 0.7
            segment.elasticity = 0.1
            self.space.add(segment)

    def add_circles_batch(self, num_circles):
        circles = []
        spread = self.rect_width * 0.45

        for _ in range(num_circles):
            radius = random.choice(self.small_circle_radii)

            x = self.center_x + random.uniform(-spread, spread)
            y = self.center_y - self.rect_height//2 + random.uniform(-50, 0)

            mass = 1.0
            moment = pymunk.moment_for_circle(mass, 0, radius)
            body = pymunk.Body(mass, moment)
            body.position = x, y

            shape = pymunk.Circle(body, radius - 0.5)
            shape.friction = 0.7
            shape.elasticity = 0.1
            shape.collision_type = 1

            self.space.add(body, shape)
            circles.append((shape, radius))

        return circles

    def create_initial_images(self):
        """Create base images and drawing objects"""
        img = Image.new('RGB', (self.width, self.height), 'white')
        mask = Image.new('L', (self.width, self.height), 0)
        mask_draw = ImageDraw.Draw(mask)
        circles_img = Image.new('RGB', (self.width, self.height), '#FFF6E9')
        circles_draw = ImageDraw.Draw(circles_img)

        return img, mask, mask_draw, circles_img, circles_draw

    def draw_decorative_rings(self, draw):
        """Draw the decorative outer, main, and inner rings"""
        # Outer ring
        outer_ring_radius = self.main_circle_radius + 20
        draw.ellipse([
            self.center_x - outer_ring_radius,
            self.center_y - outer_ring_radius,
            self.center_x + outer_ring_radius,
            self.center_y + outer_ring_radius
        ], fill='#FFF6E9', outline='#E8D0A9', width=3)

        # Main circle with gradient edge
        for i in range(5):
            draw.ellipse([
                self.center_x - self.main_circle_radius - i,
                self.center_y - self.main_circle_radius - i,
                self.center_x + self.main_circle_radius + i,
                self.center_y + self.main_circle_radius + i
            ], fill='#FFF6E9', outline='#E8D0A9', width=1)

        # Inner ring
        inner_ring_radius = self.main_circle_radius - 10
        draw.ellipse([
            self.center_x - inner_ring_radius,
            self.center_y - inner_ring_radius,
            self.center_x + inner_ring_radius,
            self.center_y + inner_ring_radius
        ], fill='#FFF6E9', outline='#E8D0A9', width=2)

    def organize_circles_by_position(self, circles):
        """Group and sort circles by their position for better color distribution"""
        circle_regions = []
        for circle, radius in circles:
            pos = circle.body.position
            angle = math.atan2(pos.y - self.center_y, pos.x - self.center_x)
            dist = math.sqrt((pos.x - self.center_x)**2 + (pos.y - self.center_y)**2)
            circle_regions.append((circle, radius, angle, dist))

        # Sort by angle and distance for more pleasing distribution
        circle_regions.sort(key=lambda x: (x[2], x[3]))
        return circle_regions

    def draw_circle_with_gradient(self, draw, pos, radius, color):
        """Draw a single circle with gradient effect"""
        for i in range(3):
            draw.ellipse([
                pos.x - radius + i,
                pos.y - radius + i,
                pos.x + radius - i,
                pos.y + radius - i
            ], fill=color, outline=self.darken_color(color))

    def draw_circles(self, circles_draw, circle_regions):
        """Draw all circles with artistic color distribution"""
        for circle, radius, angle, dist in circle_regions:
            pos = circle.body.position
            if self.is_inside_main_circle(pos.x, pos.y):
                if self.is_inside_number(pos.x, pos.y):
                    base_color = self.get_next_figure_color()
                else:
                    base_color = self.get_next_background_color()

                color = self.adjust_color(base_color, angle, dist)
                self.draw_circle_with_gradient(circles_draw, pos, radius, color)



    def generate_plate(self):
        """Main method to generate the Ishihara plate"""
        # Run physics simulation (previous code remains the same)
        circles = self.run_physics_simulation()

        # Create images and drawing objects
        img, mask, mask_draw, circles_img, circles_draw = self.create_initial_images()

        # Draw the decorative elements
        self.draw_decorative_rings(circles_draw)

        # Draw the mask for the main circle
        mask_draw.ellipse([
            self.center_x - self.main_circle_radius,
            self.center_y - self.main_circle_radius,
            self.center_x + self.main_circle_radius,
            self.center_y + self.main_circle_radius
        ], fill=255)

        # Organize and draw circles
        circle_regions = self.organize_circles_by_position(circles)
        self.draw_circles(circles_draw, circle_regions)

        # Apply mask and combine layers
        img.paste(circles_img, (0, 0), mask)

        inside_circles = [(c, r) for c, r in circles
                        if self.is_inside_main_circle(c.body.position.x, c.body.position.y)]
        return img, inside_circles

    def run_physics_simulation(self):
        """Run the physics simulation to place circles"""
        circles = []
        batch_size = 50
        batches = 0
        max_batches = 40

        while batches < max_batches:
            new_circles = self.add_circles_batch(batch_size)
            circles.extend(new_circles)

            for _ in range(20):
                self.space.step(1/60.0)

            batches += 1

        # Extra settling time
        for _ in range(120):
            self.space.step(1/60.0)

        return circles

    def adjust_color(self, color, angle, dist):
        """Adjust color based on position for more artistic variation"""
        # Convert hex to RGB
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)

        # Add subtle variation based on angle and distance
        variation = 10
        r = max(0, min(255, r + int(math.cos(angle) * variation)))
        g = max(0, min(255, g + int(math.sin(dist/100) * variation)))
        b = max(0, min(255, b + int(math.cos(dist/100) * variation)))

        return f'#{r:02x}{g:02x}{b:02x}'

    def darken_color(self, color):
        """Create slightly darker version of color for subtle edge effect"""
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)

        factor = 0.9
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)

        return f'#{r:02x}{g:02x}{b:02x}'








# api
def generate_ishihara_plate(num: int = 5):
    palette_manager = PaletteManager()
    generator = IshiharaPlateGenerator(palette_manager, num=n)
    image, circles = generator.generate_plate()
    return image, circles

