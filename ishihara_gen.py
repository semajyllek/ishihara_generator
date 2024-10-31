
from pathlib import Path


import math
import random
import numpy as np


import pygame
import pymunk
import pymunk.pygame_util

from PIL import Image, ImageDraw
from PIL import ImageColor

from typing import Optional
import random

from .color_generator import generate_ishihara_palette
from .inttogrid import generate_binary_image_with_font, get_damion_tff

# Fixed constants for circle sizes
LARGE_CIRCLE_DIAMETER = 800  # pixels
SMALL_CIRCLE_DIAMETERS = [24, 28, 32, 36, 40, 44, 48, 52]  # Wider range of sizes

GRID_SIZE = 20
FONT_SIZE = 128
DEMO_NUMBER = 5


class IshiharaPlateGenerator:
    def __init__(self, num: int = DEMO_NUMBER, font_path: Optional[str] = None):

        # creates binary grids with stylistic integer mask
        if font_path is None:
            font_path = get_damion_tff()
            
        self.bin_num = generate_binary_image_with_font(num=num, font_path=font_path, size=GRID_SIZE)
        
        # circle sizes and positions
        self.main_circle_radius = LARGE_CIRCLE_DIAMETER // 2
        self.small_circle_radii = [d // 2 for d in SMALL_CIRCLE_DIAMETERS]
        self.max_small_radius = max(self.small_circle_radii)

        self.rect_width = LARGE_CIRCLE_DIAMETER
        self.rect_height = LARGE_CIRCLE_DIAMETER

        self.width = LARGE_CIRCLE_DIAMETER + 200
        self.height = LARGE_CIRCLE_DIAMETER + 200

        self.center_x = self.width // 2
        self.center_y = self.height // 2

        # physics simulation params
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 900.0)

        self.space.collision_slop = 0.0
        self.space.collision_bias = pow(1.0 - 0.3, 60.0)
        self.space.iterations = 30

      
        # color palette 
        selected_palette = generate_ishihara_palette()
        
        self.background_colors = selected_palette['colors']['background']
        self.figure_colors = selected_palette['colors']['figure']
        self.border_color = selected_palette['colors']['border']
        self.background_base = selected_palette['colors']['background_base']
        
        self.current_bg_color_index = 0
        self.current_fg_color_index = 0

        self.create_boundary()

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
            return self.bin_num[grid_y, grid_x] == 1
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
        # Use golden ratio for more natural-looking distribution
        golden_ratio = (1 + 5 ** 0.5) / 2
        
        for i in range(num_circles):
            radius = random.choice(self.small_circle_radii)
            
            # Create a more organic distribution
            angle = i * golden_ratio * 2 * math.pi
            r = random.uniform(0, self.rect_width * 0.45)
            
            x = self.center_x + r * math.cos(angle)
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
        # Draw main background circle first
        draw.ellipse([
            self.center_x - self.main_circle_radius,
            self.center_y - self.main_circle_radius,
            self.center_x + self.main_circle_radius,
            self.center_y + self.main_circle_radius
        ], fill=self.background_base)
        
        # Draw bold black border right at the main circle's edge
        for i in range(8):  # Increased number of lines
            draw.ellipse([
                self.center_x - self.main_circle_radius - i,
                self.center_y - self.main_circle_radius - i,
                self.center_x + self.main_circle_radius + i,
                self.center_y + self.main_circle_radius + i
            ], fill=None, outline='black', width=3)  # Increased width
        
        # Inner decorative ring
        inner_ring_radius = self.main_circle_radius - 20
        draw.ellipse([
            self.center_x - inner_ring_radius,
            self.center_y - inner_ring_radius,
            self.center_x + inner_ring_radius,
            self.center_y + inner_ring_radius
        ], fill=None, outline=self.border_color, width=2)

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
        """Draw a single circle with subtle gradient effect"""
        try:
            rgb = ImageColor.getrgb(color)
            
            # Create darker and lighter versions
            darker = tuple(int(c * 0.95) for c in rgb)
            lighter = tuple(int(min(255, c * 1.05)) for c in rgb)
            
            # Main circle
            draw.ellipse([
                pos.x - radius,
                pos.y - radius,
                pos.x + radius,
                pos.y + radius
            ], fill=color)
            
            # Highlight
            highlight_radius = radius * 0.7
            draw.ellipse([
                pos.x - highlight_radius,
                pos.y - highlight_radius,
                pos.x + highlight_radius * 0.8,
                pos.y + highlight_radius * 0.8
            ], fill=lighter)
            
        except Exception as e:
            # Fallback to simple circle if gradient fails
            draw.ellipse([
                pos.x - radius,
                pos.y - radius,
                pos.x + radius,
                pos.y + radius
            ], fill=color)


    def add_subtle_texture(self, img):
        """Add subtle noise texture to the image"""
        width, height = img.size
        pixels = img.load()
        
        for x in range(width):
            for y in range(height):
                if isinstance(pixels[x, y], int):  # Handle grayscale images
                    continue
                r, g, b = pixels[x, y][:3]
                noise = random.randint(-5, 5)
                pixels[x, y] = (
                    max(0, min(255, r + noise)),
                    max(0, min(255, g + noise)),
                    max(0, min(255, b + noise))
                )


    def draw_circles(self, circles_draw, circle_regions):
        """Draw all circles with enhanced visual effects"""
        for circle, radius, angle, dist in circle_regions:
            pos = circle.body.position
            if self.is_inside_main_circle(pos.x, pos.y):
                if self.is_inside_number(pos.x, pos.y):
                    base_color = self.get_next_figure_color()
                else:
                    base_color = self.get_next_background_color()
                
                self.draw_circle_with_gradient(circles_draw, pos, radius, base_color)


    def draw_base_circle(self, draw):
        """Draw the main background circle and inner decorative ring"""
        draw.ellipse([
            self.center_x - self.main_circle_radius,
            self.center_y - self.main_circle_radius,
            self.center_x + self.main_circle_radius,
            self.center_y + self.main_circle_radius
        ], fill=self.background_base)
        
        # Inner decorative ring
        inner_ring_radius = self.main_circle_radius - 20
        draw.ellipse([
            self.center_x - inner_ring_radius,
            self.center_y - inner_ring_radius,
            self.center_x + inner_ring_radius,
            self.center_y + inner_ring_radius
        ], fill=None, outline=self.border_color, width=2)

    def create_circle_mask(self, mask_draw):
        """Create mask for the main circle"""
        mask_draw.ellipse([
            self.center_x - self.main_circle_radius,
            self.center_y - self.main_circle_radius,
            self.center_x + self.main_circle_radius,
            self.center_y + self.main_circle_radius
        ], fill=255)


    def draw_bold_border(self, draw):
        """Draw the bold black border on top of everything"""
        for i in range(8):
            draw.ellipse([
                self.center_x - self.main_circle_radius - i,
                self.center_y - self.main_circle_radius - i,
                self.center_x + self.main_circle_radius + i,
                self.center_y + self.main_circle_radius + i
            ], fill=None, outline='black', width=3)

    def get_inside_circles(self, circles):
        """Get only the circles that are inside the main circle"""
        return [(c, r) for c, r in circles 
                if self.is_inside_main_circle(c.body.position.x, c.body.position.y)]


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
    

    def generate_plate(self):
        # Adjust physics parameters for better spacing
        self.space.iterations = 50
        self.space.collision_slop = 0.001
        
        # Run physics simulation to place circles
        circles = self.run_physics_simulation()

        # Create images and drawing objects
        img, mask, mask_draw, circles_img, circles_draw = self.create_initial_images()
        
        # Draw the main circle and inner ring
        self.draw_base_circle(circles_draw)
        self.create_circle_mask(mask_draw)
        
        # Draw the decorative rings
        circle_regions = self.organize_circles_by_position(circles)
        self.draw_circles(circles_draw, circle_regions)
        
        # Add texture before final composition
        self.add_subtle_texture(circles_img)
        
        # Combine all images
        img.paste(circles_img, (0, 0), mask)

        # Draw the bold black border
        self.draw_bold_border(ImageDraw.Draw(img))
        
        # Get only the circles that are inside the main circle
        inside_circles = self.get_inside_circles(circles)

        return img, inside_circles


# api
def generate_ishihara_plate(num: int = 5):
    generator = IshiharaPlateGenerator(num=num)
    image, circles = generator.generate_plate()
    return image, circles



if __name__ == '__main__':
    # Each time you generate a plate, it will use fresh, harmonious colors
    n = 7
    image, circles = generate_ishihara_plate(7)
    image.show()

