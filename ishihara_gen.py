
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
        
        self.border_color = selected_palette['colors']['border']
        self.background_base = selected_palette['colors']['background_base']
        
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

    
    def setup_number_transform(self):
        """Pre-compute coordinate transformation constants"""
        # Adjust number size to be more prominent
        self.number_width = self.main_circle_radius * 1.0  # Increased from 0.8
        self.number_height = self.main_circle_radius * 1.0
        # Center the number
        self.number_x = self.center_x - self.number_width/2
        self.number_y = self.center_y - self.number_height/2


    def get_circle_sizes(self):
        """Define circle sizes matching sample image distribution"""
        # Sizes in pixels diameter - more gradual progression
        sizes = [
            35,  # Largest - extremely rare
            26,  # Very large - very rare
            22,  # Large-medium - uncommon
            18,  # Medium - very common
            15,  # Medium-small - most common
            12,  # Small - very common
            10,  # Very small - common
            8    # Tiny - for filling gaps
        ]
        # Weights based on sample image analysis
        self.size_weights = [0.01, 0.012, 0.05, 0.17, 0.26, 0.18, 0.08, 0.03]  # Adds to 1.0
        return [s//2 for s in sizes]  # Convert to radii


    def find_number_bounds(self):
        """Calculate the bounds of the number in world coordinates"""
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')
        
        for y in range(self.bin_num.shape[0]):
            for x in range(self.bin_num.shape[1]):
                if self.bin_num[y][x]:
                    world_x = self.number_x + (x * self.number_width / self.bin_num.shape[1])
                    world_y = self.number_y + (y * self.number_height / self.bin_num.shape[0])
                    min_x = min(min_x, world_x)
                    max_x = max(max_x, world_x)
                    min_y = min(min_y, world_y)
                    max_y = max(max_y, world_y)
        
        return min_x, max_x, min_y, max_y

    def generate_initial_positions(self, min_x, max_x, min_y, max_y, grid_size):
        """Generate initial grid of positions with offset rows"""
        positions = []
        for y in np.arange(min_y, max_y, grid_size):
            row_offset = (grid_size / 2) * ((int(y / grid_size) % 2))
            for x in np.arange(min_x, max_x, grid_size):
                if self.is_inside_number(x + row_offset, y):
                    positions.append((x + row_offset, y))
        return positions
    


    def find_edge_points(self):
        """Find points along the edge of the number"""
        edge_points = []
        for y in range(1, self.bin_num.shape[0] - 1):
            for x in range(1, self.bin_num.shape[1] - 1):
                if self.bin_num[y][x]:
                    # Check if this is an edge pixel by looking at neighbors
                    is_edge = False
                    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                        if not self.bin_num[y + dy][x + dx]:
                            is_edge = True
                            break
                            
                    if is_edge:
                        world_x = self.number_x + (x * self.number_width / self.bin_num.shape[1])
                        world_y = self.number_y + (y * self.number_height / self.bin_num.shape[0])
                        edge_points.append((world_x, world_y))
        return edge_points

    def sort_positions_by_edge_proximity(self, positions, edge_points, grid_size):
        """Sort positions into edge and interior groups"""
        edge_positions = []
        interior_positions = []
        
        for pos in positions:
            min_edge_dist = min(math.sqrt((pos[0] - ex)**2 + (pos[1] - ey)**2) 
                            for ex, ey in edge_points)
            if min_edge_dist < grid_size * 2:
                edge_positions.append(pos)
            else:
                interior_positions.append(pos)
                
        random.shuffle(edge_positions)
        random.shuffle(interior_positions)
        return edge_positions + interior_positions
    




    def get_radius_options(self, x, y, edge_points, grid_size):
        """Get appropriate radii and weights based on position"""
        min_edge_dist = min(math.sqrt((x - ex)**2 + (y - ey)**2) 
                        for ex, ey in edge_points)


        if min_edge_dist < grid_size * 2:
            available_radii = self.get_circle_sizes()[-4:]  # Limit to smaller radii near edges
            weights = self.size_weights[-4:]
        else:
            available_radii = self.get_circle_sizes()
            weights = self.size_weights
        
        weights = [w/sum(weights) for w in weights]
        return available_radii, weights



    def generate_new_positions(self, x, y, radius, spacing):
        """Generate new candidate positions around a placed circle"""
        new_positions = []
        for angle in range(0, 360, 30):
            rad = math.radians(angle)
            for dist_factor in [1.1, 1.3]:
                new_x = x + math.cos(rad) * (radius * 2 + spacing) * dist_factor
                new_y = y + math.sin(rad) * (radius * 2 + spacing) * dist_factor
                if self.is_inside_number(new_x, new_y):
                    new_positions.append((new_x, new_y))
        return new_positions
    


    def add_circles_to_number(self, target_circles=1000):
        """Fill number with dense packing while following contours"""
        circles = []
        spacing = 1.0
        
        # Get number bounds
        min_x, max_x, min_y, max_y = self.find_number_bounds()
        
        # Create initial grid of positions
        grid_size = min(self.get_circle_sizes()) * 1.8
        positions = self.generate_initial_positions(min_x, max_x, min_y, max_y, grid_size)
        
        # Find edge points and sort positions
        edge_points = self.find_edge_points()
        positions = self.sort_positions_by_edge_proximity(positions, edge_points, grid_size)
        
        # Place circles
        while positions and len(circles) < target_circles:
            x, y = positions.pop(0)
            
            # Get appropriate radii for this position
            available_radii, weights = self.get_radius_options(x, y, edge_points, grid_size)
            
            # Try to place circle
            for radius in random.choices(available_radii, weights=weights, k=len(available_radii)):
                if self.try_place_circle(x, y, radius, spacing):
                    # Create and add circle
                    shape = self.create_physics_circle(x, y, radius)
                    circles.append((shape, radius))
                    
                    # Generate new positions around this circle
                    new_positions = self.generate_new_positions(x, y, radius, spacing)
                    positions.extend(new_positions)
                    break
        
        # Minimal settling
        for _ in range(10):
            self.space.step(1/60.0)
        
        return circles

    # Supporting methods that remain the same
    def try_place_circle(self, x, y, radius, spacing=1.0):
        """Check if a circle can be placed at the given position"""
        if not self.is_inside_number(x, y):
            return False
        
        for shape in self.space.shapes:
            if isinstance(shape, pymunk.Circle):
                dist = math.sqrt((x - shape.body.position.x)**2 + 
                            (y - shape.body.position.y)**2)
                if dist < (radius + shape.radius + spacing):
                    return False
        return True
    

    def create_physics_circle(self, x, y, radius):
        """Create and add a physics circle with more stability"""
        body = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0, radius))
        body.position = (x, y)
        shape = pymunk.Circle(body, radius)
        shape.friction = 0.9
        shape.elasticity = 0.0
        self.space.add(body, shape)
        return shape






    def run_physics_simulation(self):
        """Number filling only"""
        # No gravity needed for placement
        self.space.gravity = (0.0, 0.0)
        self.space.damping = 1.0
        
        # Create number boundary
        self.create_number_boundary()
        print("Created number boundary")
        
        # Fill number region
        number_circles = self.add_circles_to_number()
        
        if not number_circles:
            print("Failed to place any circles!")
        
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
        
        # Shuffle the order to avoid patterns
        random.shuffle(figure_circles)
        
        # Draw figure circles with distinct variations
        base_colors = [
            '#FF6B4D',  # Coral
            '#FF8463',  # Light coral
            '#FF4D27',  # Bright orange-red
            '#E84D1C',  # Deep orange
            '#FF5533',  # True orange-red
        ]
        
        for i, (circle, radius) in enumerate(figure_circles):
            # Select color randomly for each circle
            color = random.choice(base_colors)
            self.draw_circle_with_gradient(circles_draw, circle.body.position, radius, color)



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
        
        # Use our improved drawing method
        self.draw_circles(circles_draw, circles)
        
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

