
from pathlib import Path


import math
import random
import numpy as np


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
        
        # Pre-compute the transformed coordinates for number checking
        self.setup_number_transform()

        self.create_number_boundary()


    def is_inside_main_circle(self, x, y):
        """Check if a point is inside the main circle"""
        return (x - self.center_x)**2 + (y - self.center_y)**2 <= self.main_circle_radius**2
    
    def is_inside_number(self, x, y):
        """Check if a point is inside the number area"""
        # Convert world coordinates to grid coordinates
        grid_x = (x - self.number_x) / self.number_width * self.bin_num.shape[1]
        grid_y = (y - self.number_y) / self.number_height * self.bin_num.shape[0]
        
        # Check if inside the number area
        return 0 <= grid_x < self.bin_num.shape[1] and 0 <= grid_y < self.bin_num.shape[0] and self.bin_num[int(grid_y)][int(grid_x)]

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

    
    def fill_background(self, existing_circles):
        """Fill the background using physics simulation"""
        circles = []
        radii = sorted(self.small_circle_radii, reverse=True)
        
        # Use golden ratio spiral for initial placement
        max_attempts = 2000
        golden_angle = math.pi * (3 - math.sqrt(5))
        
        size_distribution = {r: 0 for r in radii}
        target_per_size = max_attempts / len(radii)
        
        for i in range(max_attempts):
            # Choose radius ensuring good distribution
            available_radii = [r for r in radii if size_distribution[r] < target_per_size * 1.2]
            if not available_radii:
                available_radii = radii
            radius = random.choice(available_radii)
            
            # Generate position using golden angle spiral
            r = math.sqrt(i / max_attempts) * (self.main_circle_radius - radius)
            theta = i * golden_angle
            x = self.center_x + r * math.cos(theta)
            y = self.center_y + r * math.sin(theta)
            
            # Check position validity
            if not self.is_inside_main_circle(x, y) or self.is_inside_number(x, y):
                continue
            
            # Check overlap
            overlaps = False
            for shape in self.space.shapes:
                if isinstance(shape, pymunk.Circle):
                    dist = math.sqrt((x - shape.body.position.x)**2 + 
                                (y - shape.body.position.y)**2)
                    if dist < (radius + shape.radius + 1):
                        overlaps = True
                        break
            
            if not overlaps:
                body = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0, radius))
                body.position = (x, y)
                shape = pymunk.Circle(body, radius)
                shape.friction = 0.7
                shape.elasticity = 0.1
                self.space.add(body, shape)
                circles.append((shape, radius))
                size_distribution[radius] += 1
                
                # Mini physics step
                for _ in range(10):
                    self.space.step(1/60.0)
        
        return circles


    def setup_number_transform(self):
        """Pre-compute coordinate transformation constants"""
        # Adjust number size to be more prominent
        self.number_width = self.main_circle_radius * 1.0  # Increased from 0.8
        self.number_height = self.main_circle_radius * 1.0
        # Center the number
        self.number_x = self.center_x - self.number_width/2
        self.number_y = self.center_y - self.number_height/2




    def get_circle_sizes(self):
        """Define 5 circle sizes with appropriate weights based on sample"""
        # Sizes in pixels diameter
        sizes = [
            40,  # Largest - rare
            32,  # Large - occasional
            24,  # Medium - common
            16,  # Small - very common
            12   # Smallest - very common
        ]
        # Weights determine how often each size is chosen
        self.size_weights = [0.05, 0.15, 0.25, 0.30, 0.25]  # Adds to 1.0
        return [s//2 for s in sizes]  # Convert to radii

    def add_circles_to_number(self, target_circles=1000):  # Increased target for better filling
        """Fill number completely with circles"""
        circles = []
        radii = self.get_circle_sizes()
        spacing = 0.5  # Minimal spacing between circles
        
        def try_place_circle(x, y, radius):
            """Check if a circle can be placed at position"""
            if not self.is_inside_number(x, y):
                return False
                
            # Check overlap with existing circles
            for shape in self.space.shapes:
                if isinstance(shape, pymunk.Circle):
                    dist = math.sqrt((x - shape.body.position.x)**2 + 
                                (y - shape.body.position.y)**2)
                    if dist < (radius + shape.radius + spacing):
                        return False
            return True
        
        def get_candidate_positions(current_x, current_y, radius):
            """Get positions around current point"""
            positions = []
            # Try positions in concentric circles
            for r in np.arange(radius*2, radius*4, radius):
                for angle in range(0, 360, 10):  # Every 10 degrees
                    rad = math.radians(angle)
                    x = current_x + r * math.cos(rad)
                    y = current_y + r * math.sin(rad)
                    positions.append((x, y))
            random.shuffle(positions)
            return positions
        
        # Start from center of number
        number_center_x = self.number_x + self.number_width/2
        number_center_y = self.number_y + self.number_height/2
        
        # Keep track of positions to try
        positions_to_try = [(number_center_x, number_center_y)]
        attempted_positions = set()
        
        while positions_to_try and len(circles) < target_circles:
            current_x, current_y = positions_to_try.pop(0)
            pos_key = (round(current_x), round(current_y))
            
            if pos_key in attempted_positions:
                continue
                
            attempted_positions.add(pos_key)
            
            # Try each radius based on weighted distribution
            radius = random.choices(radii, weights=self.size_weights, k=1)[0]
            
            if try_place_circle(current_x, current_y, radius):
                # Place circle
                body = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0, radius))
                body.position = (current_x, current_y)
                shape = pymunk.Circle(body, radius)
                shape.friction = 0.7
                shape.elasticity = 0.1
                self.space.add(body, shape)
                circles.append((shape, radius))
                
                # Add new positions to try around this circle
                new_positions = get_candidate_positions(current_x, current_y, radius)
                positions_to_try.extend(new_positions)
                
                # Periodically shuffle positions for better distribution
                if len(circles) % 10 == 0:
                    random.shuffle(positions_to_try)
            
            # If we can't place at current radius, try smaller ones
            else:
                for smaller_radius in [r for r in radii if r < radius]:
                    if try_place_circle(current_x, current_y, smaller_radius):
                        body = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0, smaller_radius))
                        body.position = (current_x, current_y)
                        shape = pymunk.Circle(body, smaller_radius)
                        shape.friction = 0.7
                        shape.elasticity = 0.1
                        self.space.add(body, shape)
                        circles.append((shape, smaller_radius))
                        break
        
        return circles

    def run_physics_simulation(self):
        """Number filling only"""
        self.space.gravity = (0.0, 0.0)  # No gravity needed for placement
        self.space.damping = 1.0  # Maximum damping
        
        # Create number boundary
        self.create_number_boundary()
        
        # Fill number region
        number_circles = self.add_circles_to_number()
        
        return number_circles


    def create_number_boundary(self):
        """Create physical boundaries for both the number and main circle"""
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.space.add(body)
        
        # First create the main circle boundary
        num_segments = 32  # Number of segments to approximate circle
        for i in range(num_segments):
            angle1 = 2 * math.pi * i / num_segments
            angle2 = 2 * math.pi * (i + 1) / num_segments
            p1 = (self.center_x + math.cos(angle1) * self.main_circle_radius,
                self.center_y + math.sin(angle1) * self.main_circle_radius)
            p2 = (self.center_x + math.cos(angle2) * self.main_circle_radius,
                self.center_y + math.sin(angle2) * self.main_circle_radius)
            segment = pymunk.Segment(body, p1, p2, 1.0)
            segment.friction = 0.7
            segment.elasticity = 0.1
            self.space.add(segment)
        
        # Then create the number boundary
        points = []
        for y in range(self.bin_num.shape[0]):
            for x in range(self.bin_num.shape[1]):
                if self.bin_num[y][x]:
                    # Get neighboring cells
                    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.bin_num.shape[1] and 
                            0 <= ny < self.bin_num.shape[0]):
                            if not self.bin_num[ny][nx]:
                                # This is a boundary cell
                                world_x = self.number_x + (x * self.number_width / self.bin_num.shape[1])
                                world_y = self.number_y + (y * self.number_height / self.bin_num.shape[0])
                                points.append((world_x, world_y))
        
        # Create segments for the number boundary
        if points:
            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]
                segment = pymunk.Segment(body, p1, p2, 1.0)
                segment.friction = 0.7
                segment.elasticity = 0.1
                self.space.add(segment)
        
        return body


  

    def check_coverage(self, circles):
        """Improved coverage checking with grid-based sampling"""
        grid_resolution = 50
        coverage_points = 0
        total_points = 0
        
        # Create a grid of sample points
        for y in range(self.bin_num.shape[0]):
            for x in range(self.bin_num.shape[1]):
                if self.bin_num[y][x]:
                    total_points += 1
                    
                    # Convert grid position to world coordinates
                    world_x = self.number_x + (x + 0.5) * (self.number_width / self.bin_num.shape[1])
                    world_y = self.number_y + (y + 0.5) * (self.number_height / self.bin_num.shape[0])
                    
                    # Check if point is covered by any circle
                    for circle, radius in circles:
                        pos = circle.body.position
                        dist_sq = (world_x - pos.x)**2 + (world_y - pos.y)**2
                        if dist_sq <= (radius * 0.95)**2:  # 95% of radius to ensure overlap
                            coverage_points += 1
                            break
        
        return coverage_points / total_points if total_points > 0 else 0

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

    def draw_circles(self, circles_draw, circle_regions):
        """Improved color distribution while maintaining simplicity"""
        # Separate circles into figure and background groups
        figure_circles = []
        background_circles = []
        
        for circle, radius in circle_regions:
            pos = circle.body.position
            if not self.is_inside_main_circle(pos.x, pos.y):
                continue
                
            if self.is_inside_number(pos.x, pos.y):
                figure_circles.append((circle, radius))
            else:
                background_circles.append((circle, radius))
        
        # Shuffle both groups
        random.shuffle(figure_circles)
        random.shuffle(background_circles)
        
        # Draw background circles with varying colors
        for i, (circle, radius) in enumerate(background_circles):
            color_index = (i * 3) % len(self.background_colors)  # Skip colors for more variety
            color = self.background_colors[color_index]
            self.draw_circle_with_gradient(circles_draw, circle.body.position, radius, color)
        
        # Draw figure circles with varying colors
        for i, (circle, radius) in enumerate(figure_circles):
            color_index = (i * 3) % len(self.figure_colors)  # Skip colors for more variety
            color = self.figure_colors[color_index]
            self.draw_circle_with_gradient(circles_draw, circle.body.position, radius, color)

    def color_difference(self, color1, color2):
        """Calculate approximate color difference in RGB space"""
        # Convert hex to RGB
        r1 = int(color1[1:3], 16)
        g1 = int(color1[3:5], 16)
        b1 = int(color1[5:7], 16)
        
        r2 = int(color2[1:3], 16)
        g2 = int(color2[3:5], 16)
        b2 = int(color2[5:7], 16)
        
        # Calculate Euclidean distance in RGB space
        return math.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)

    def get_next_background_color(self, last_color=None):
        """Get next background color ensuring variety"""
        if last_color is not None:
            # Try to pick a color different enough from the last one
            attempts = 0
            while attempts < len(self.background_colors):
                color = self.background_colors[self.current_bg_color_index]
                if self.color_difference(color, last_color) > 30:  # threshold for difference
                    break
                self.current_bg_color_index = (self.current_bg_color_index + 1) % len(self.background_colors)
                attempts += 1
        else:
            color = self.background_colors[self.current_bg_color_index]
        
        self.current_bg_color_index = (self.current_bg_color_index + 1) % len(self.background_colors)
        return color

    def get_next_figure_color(self, last_color=None):
        """Get next figure color ensuring variety"""
        if last_color is not None:
            # Try to pick a color different enough from the last one
            attempts = 0
            while attempts < len(self.figure_colors):
                color = self.figure_colors[self.current_fg_color_index]
                if self.color_difference(color, last_color) > 30:  # threshold for difference
                    break
                self.current_fg_color_index = (self.current_fg_color_index + 1) % len(self.figure_colors)
                attempts += 1
        else:
            color = self.figure_colors[self.current_fg_color_index]
        
        self.current_fg_color_index = (self.current_fg_color_index + 1) % len(self.figure_colors)
        return color

    def draw_bold_border(self, draw):
        """Draw the bold black border on top of everything"""
        for i in range(8):
            draw.ellipse([
                self.center_x - self.main_circle_radius - i,
                self.center_y - self.main_circle_radius - i,
                self.center_x + self.main_circle_radius + i,
                self.center_y + self.main_circle_radius + i
            ], fill=None, outline='black', width=3)

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

