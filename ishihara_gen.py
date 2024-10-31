
from pathlib import Path


import math
import random
import numpy as np

from functools import lru_cache
from scipy.ndimage import binary_dilation


import pymunk
import pymunk.pygame_util

from PIL import Image, ImageDraw
from PIL import ImageColor

from typing import Optional
import random

from .color_generator import generate_ishihara_palette
from .int_to_grid import DigitRenderer, get_tff

# Fixed constants for circle sizes
LARGE_CIRCLE_DIAMETER = 800  # pixels
SMALL_CIRCLE_DIAMETERS = [24, 28, 32, 36, 40, 44, 48, 52]  # Wider range of sizes

GRID_SIZE = 20
FONT_SIZE = 128
DEMO_NUMBER = 5

FONT_URL = 'https://www.1001fonts.com/download/niconne.zip'



class IshiharaPlateGenerator:
    def __init__(self, num: int = DEMO_NUMBER, font_zip_url: Optional[str] = FONT_URL):


        # creates binary grids with stylistic integer mask
        self.renderer = DigitRenderer(font_size=FONT_SIZE, font_path=get_tff(font_zip_url))
        self.bin_num = self.renderer.digit_to_grid(digit=num, size=GRID_SIZE)
        
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


        # Pre-compute the dilated number mask for faster edge checking
        self.dilated_mask = binary_dilation(self.bin_num, iterations=2)
        
        # Pre-compute the transformed coordinates for number checking
        self.setup_number_transform()

        self.create_boundary()

    def setup_number_transform(self):
        """Pre-compute coordinate transformation constants"""
        self.number_width = self.main_circle_radius * 1.4
        self.number_height = self.main_circle_radius * 1.4
        self.number_x = self.center_x - self.number_width/2
        self.number_y = self.center_y - self.number_height/2 - self.main_circle_radius * 0.1


    def get_next_background_color(self):
        color = self.background_colors[self.current_bg_color_index]
        self.current_bg_color_index = (self.current_bg_color_index + 1) % len(self.background_colors)
        return color

    def get_next_figure_color(self):
        color = self.figure_colors[self.current_fg_color_index]
        self.current_fg_color_index = (self.current_fg_color_index + 1) % len(self.figure_colors)
        return color
    
    @lru_cache(maxsize=1024)
    def is_inside_main_circle(self, x, y):
        dx = x - self.center_x
        dy = y - self.center_y
        return dx*dx + dy*dy <= self.main_circle_radius * self.main_circle_radius
    

    def is_inside_number(self, x, y):
        """Optimized number boundary checking"""
        # Convert to grid coordinates using pre-computed transforms
        grid_x = int((x - self.number_x) / self.x_scale)
        grid_y = int((y - self.number_y) / self.y_scale)
        
        if 0 <= grid_x < self.bin_num.shape[1] and 0 <= grid_y < self.bin_num.shape[0]:
            return self.bin_num[grid_y, grid_x] == 1
        return False
  

    def add_circles_batch(self, num_circles):
        """Modified circle placement that accounts for ring space"""
        circles = []
        golden_ratio = (1 + 5 ** 0.5) / 2
        
        for i in range(num_circles):
            radius = random.choice(self.small_circle_radii)
            
            # Add extra space for the white ring
            ring_space = radius * 0.08  # Same as visual ring thickness
            physics_radius = radius + ring_space
            
            angle = i * golden_ratio * 2 * math.pi
            r = random.uniform(0, self.rect_width * 0.45)
            
            x = self.center_x + r * math.cos(angle)
            y = self.center_y - self.rect_height//2 + random.uniform(-50, 0)
            
            mass = 1.0
            moment = pymunk.moment_for_circle(mass, 0, physics_radius)
            body = pymunk.Body(mass, moment)
            body.position = x, y
            
            # Use the larger radius for collision detection
            shape = pymunk.Circle(body, physics_radius)
            shape.friction = 0.7
            shape.elasticity = 0.1
            shape.collision_type = 1
            
            self.space.add(body, shape)
            # Store the visual radius (without ring space) for drawing
            circles.append((shape, radius))
        
        return circles

    def draw_circle_with_gradient(self, draw, pos, radius, color):
        """Draw a non-overlapping circle with white ring"""
        try:
            # White background circle (ring)
            draw.ellipse([
                pos.x - radius - radius * 0.08,  # Add 8% for ring
                pos.y - radius - radius * 0.08,
                pos.x + radius + radius * 0.08,
                pos.y + radius + radius * 0.08
            ], fill='white')
            
            # Main colored circle
            draw.ellipse([
                pos.x - radius,
                pos.y - radius,
                pos.x + radius,
                pos.y + radius
            ], fill=color)
            
        except Exception as e:
            # Fallback to simple circle if drawing fails
            draw.ellipse([
                pos.x - radius,
                pos.y - radius,
                pos.x + radius,
                pos.y + radius
            ], fill=color)

    def run_physics_simulation(self):
        """Optimized physics simulation"""
        circles = []
        batch_size = 100  # Larger batches for better performance
        total_circles_needed = 2000  # Target number of circles
        settling_steps = 20  # Reduced settling steps
        
        # Pre-allocate space for circle positions
        circle_positions = np.zeros((total_circles_needed, 2))
        current_index = 0
        
        while current_index < total_circles_needed:
            new_circles = self.add_circles_batch(batch_size)
            
            # Batch physics updates
            for _ in range(settling_steps):
                self.space.step(1/30.0)  # Reduced precision for speed
            
            # Store only circles that are within bounds
            for circle, radius in new_circles:
                pos = circle.body.position
                if self.is_inside_main_circle(pos.x, pos.y):
                    circles.append((circle, radius))
                    circle_positions[current_index] = [pos.x, pos.y]
                    current_index += 1
                    if current_index >= total_circles_needed:
                        break
        
        # Final quick settling
        for _ in range(60):  # Reduced final settling time
            self.space.step(1/30.0)
        
        return circles


    def is_near_number_edge(self, x, y, threshold=2):
        """
        Check if a point is near the edge of the number.
        """
        # Check points in a small radius around the given point
        for dx in range(-threshold, threshold + 1):
            for dy in range(-threshold, threshold + 1):
                if self.is_inside_number(x + dx, y + dy) != self.is_inside_number(x, y):
                    return True
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
        """Optimized plate generation"""
        self.space.iterations = 20  # Reduced iterations
        
        # Run physics simulation
        circles = self.run_physics_simulation()
        
        # Create images
        img = Image.new('RGB', (self.width, self.height), 'white')
        mask = Image.new('L', (self.width, self.height), 0)
        circles_img = Image.new('RGB', (self.width, self.height), self.background_base)
        
        # Use faster drawing contexts
        mask_draw = ImageDraw.Draw(mask)
        circles_draw = ImageDraw.Draw(circles_img)
        
        # Draw base elements
        self.draw_base_circle(circles_draw)
        self.create_circle_mask(mask_draw)
        
        # Batch process circles by color
        figure_circles = []
        background_circles = []
        
        for circle, radius in circles:
            pos = circle.body.position
            if self.is_inside_number(pos.x, pos.y):
                figure_circles.append((pos, radius))
            else:
                background_circles.append((pos, radius))
        
        # Draw circles by color groups
        for circles_group, color in [
            (background_circles, self.get_next_background_color()),
            (figure_circles, self.get_next_figure_color())
        ]:
            for pos, radius in circles_group:
                self.draw_circle_with_gradient(circles_draw, pos, radius, color)
        
        # Composite images
        img.paste(circles_img, (0, 0), mask)
        
        # Draw border
        final_draw = ImageDraw.Draw(img)
        self.draw_bold_border(final_draw)
        
        return img, circles


# api
def generate_ishihara_plate(num: int = 5, font_zip_url: str = FONT_URL):
    generator = IshiharaPlateGenerator(num=num, font_zip_url=font_zip_url)
    image, circles = generator.generate_plate()
    return image, circles



if __name__ == '__main__':
    # Each time you generate a plate, it will use fresh, harmonious colors
    n = 7
    image, circles = generate_ishihara_plate(7)
    image.show()

