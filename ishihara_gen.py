
from pathlib import Path


import math
import random
import numpy as np

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
    
    def is_inside_main_circle(self, x, y):
        dx = x - self.center_x
        dy = y - self.center_y
        return dx*dx + dy*dy <= self.main_circle_radius * self.main_circle_radius
    

    def is_inside_number(self, x, y):
        """
        Convert coordinates to number grid space
        """
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
    

    def add_circles_batch(self, num_circles):
        """Enhanced circle placement with better coverage"""
        circles = []
        golden_ratio = (1 + 5 ** 0.5) / 2
        
        # Add more variation in circle placement
        for i in range(num_circles):
            radius = random.choice(self.small_circle_radii)
            ring_space = radius * 0.08
            physics_radius = radius + ring_space
            
            # Improved spiral placement with random offset
            angle = i * golden_ratio * 2 * math.pi
            base_r = (i / num_circles) * self.rect_width * 0.45
            r = base_r + random.uniform(-20, 20)  # Add some randomness
            
            # Try multiple positions for better coverage
            best_x = self.center_x
            best_y = self.center_y
            best_coverage = 0
            
            for _ in range(3):  # Try 3 different positions
                test_x = self.center_x + r * math.cos(angle + random.uniform(-0.2, 0.2))
                test_y = self.center_y - self.rect_height//2 + random.uniform(-50, 50)
                
                coverage = self.evaluate_position_coverage(test_x, test_y, radius)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_x = test_x
                    best_y = test_y
            
            mass = 1.0
            moment = pymunk.moment_for_circle(mass, 0, physics_radius)
            body = pymunk.Body(mass, moment)
            body.position = best_x, best_y
            
            shape = pymunk.Circle(body, physics_radius)
            shape.friction = 0.7
            shape.elasticity = 0.1
            shape.collision_type = 1
            
            self.space.add(body, shape)
            circles.append((shape, radius))
        
        return circles

    def evaluate_position_coverage(self, x, y, radius):
        """Evaluate how well a position contributes to number coverage"""
        score = 0
        check_points = 8  # Number of points to check around the circle
        
        for i in range(check_points):
            angle = (i / check_points) * 2 * math.pi
            check_x = x + radius * math.cos(angle)
            check_y = y + radius * math.sin(angle)
            
            # Check if point is inside number and not already well-covered
            if self.is_inside_number(check_x, check_y):
                score += 1
                # Bonus for being near edge
                if self.is_near_number_edge(check_x, check_y):
                    score += 0.5
            
        return score

    def calculate_settling_iterations(self, circles):
        """Calculate appropriate number of physics simulation iterations"""
        # Base number of iterations
        base_iterations = 30
        
        # Count circles near edges and in critical areas
        edge_circles = 0
        critical_circles = 0
        
        for circle, radius in circles:
            pos = circle.body.position
            
            # Check if circle is near edge
            if self.is_near_number_edge(pos.x, pos.y):
                edge_circles += 1
            
            # Check if circle is in a critical area (intersections or tight spaces)
            if self.is_critical_position(pos.x, pos.y):
                critical_circles += 1
        
        # Add iterations based on circle positions
        edge_factor = (edge_circles / len(circles)) if circles else 0
        critical_factor = (critical_circles / len(circles)) if circles else 0
        
        # Calculate total iterations
        total_iterations = base_iterations
        total_iterations += int(edge_factor * 20)  # Up to 20 extra iterations for edge circles
        total_iterations += int(critical_factor * 15)  # Up to 15 extra iterations for critical circles
        
        return total_iterations

    def is_critical_position(self, x, y):
        """Check if position is in a critical area requiring more careful settling"""
        critical = False
        check_radius = self.max_small_radius * 2
        
        # Check surrounding points for tight spaces or intersections
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            check_x = x + check_radius * math.cos(rad)
            check_y = y + check_radius * math.sin(rad)
            
            # Position is critical if it's near intersections of number regions
            number_state = self.is_inside_number(x, y)
            check_state = self.is_inside_number(check_x, check_y)
            
            if number_state != check_state:
                critical = True
                break
            
            # Also critical if near the main circle edge
            dist_to_center = math.sqrt((x - self.center_x)**2 + (y - self.center_y)**2)
            if abs(dist_to_center - self.main_circle_radius) < check_radius:
                critical = True
                break
        
        return critical
    
    def run_physics_simulation(self):
        """Modified physics simulation with proper iteration calculation"""
        circles = []
        batch_size = 50
        max_batches = 50
        coverage_threshold = 0.85
        
        uncovered_regions = set()
        
        for batch in range(max_batches):
            # Analyze current coverage
            coverage = self.analyze_coverage(circles)
            if coverage >= coverage_threshold:
                break
            
            # Add circles based on current coverage
            if batch > max_batches // 2:
                new_circles = self.add_targeted_circles_batch(batch_size // 2, uncovered_regions)
            else:
                new_circles = self.add_circles_batch(batch_size)
            
            circles.extend(new_circles)
            
            # Calculate and run settling iterations
            settling_iterations = self.calculate_settling_iterations(new_circles)
            for _ in range(settling_iterations):
                self.space.step(1/60.0)
                self.apply_coverage_forces(new_circles)
            
            # Update uncovered regions
            uncovered_regions = self.find_uncovered_regions(circles)
        
        # Final settling phase
        final_iterations = 60 + len(circles) // 10
        for _ in range(final_iterations):
            self.space.step(1/60.0)
        
        return circles


    def analyze_coverage(self, circles):
        """Analyze current coverage of the number"""
        grid_size = 20
        covered_points = set()
        total_points = 0
        
        for y in range(grid_size):
            for x in range(grid_size):
                if self.bin_num[y, x]:
                    total_points += 1
                    point_x = self.number_x + (x + 0.5) * (self.number_width / grid_size)
                    point_y = self.number_y + (y + 0.5) * (self.number_height / grid_size)
                    
                    for circle, radius in circles:
                        pos = circle.body.position
                        if (pos.x - point_x) ** 2 + (pos.y - point_y) ** 2 <= radius ** 2:
                            covered_points.add((x, y))
                            break
        
        return len(covered_points) / total_points if total_points > 0 else 0

    def find_uncovered_regions(self, circles):
        """Find regions that need more coverage"""
        uncovered = set()
        grid_size = 20
        
        for y in range(grid_size):
            for x in range(grid_size):
                if self.bin_num[y, x]:
                    point_x = self.number_x + (x + 0.5) * (self.number_width / grid_size)
                    point_y = self.number_y + (y + 0.5) * (self.number_height / grid_size)
                    
                    is_covered = False
                    for circle, radius in circles:
                        pos = circle.body.position
                        if (pos.x - point_x) ** 2 + (pos.y - point_y) ** 2 <= radius ** 2:
                            is_covered = True
                            break
                    
                    if not is_covered:
                        uncovered.add((point_x, point_y))
        
        return uncovered

    def add_targeted_circles_batch(self, num_circles, uncovered_regions):
        """Add circles specifically targeting uncovered regions"""
        circles = []
        uncovered_list = list(uncovered_regions)
        
        for _ in range(num_circles):
            if not uncovered_list:
                break
                
            target = random.choice(uncovered_list)
            radius = random.choice(self.small_circle_radii)
            ring_space = radius * 0.08
            physics_radius = radius + ring_space
            
            mass = 1.0
            moment = pymunk.moment_for_circle(mass, 0, physics_radius)
            body = pymunk.Body(mass, moment)
            
            # Add some randomness to prevent exact stacking
            body.position = (
                target[0] + random.uniform(-radius/2, radius/2),
                target[1] + random.uniform(-radius/2, radius/2)
            )
            
            shape = pymunk.Circle(body, physics_radius)
            shape.friction = 0.7
            shape.elasticity = 0.1
            shape.collision_type = 1
            
            self.space.add(body, shape)
            circles.append((shape, radius))
        
        return circles

    def apply_coverage_forces(self, circles):
        """Apply forces to improve coverage"""
        for circle, radius in circles:
            pos = circle.body.position
            if self.is_inside_number(pos.x, pos.y):
                # Apply gentle force toward uncovered nearby areas
                for angle in range(0, 360, 45):
                    rad = math.radians(angle)
                    check_x = pos.x + radius * 2 * math.cos(rad)
                    check_y = pos.y + radius * 2 * math.sin(rad)
                    
                    if self.is_inside_number(check_x, check_y):
                        force_mag = 50.0
                        circle.body.apply_force_at_local_point((
                            math.cos(rad) * force_mag,
                            math.sin(rad) * force_mag
                        ))

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
        """Draw circles with simpler but more diverse color distribution"""
        # Reset color indices at start
        self.current_bg_color_index = 0
        self.current_fg_color_index = 0
        
        # Shuffle the circle order before drawing to break up patterns
        random.shuffle(circle_regions)
        
        for circle, radius in circle_regions:
            pos = circle.body.position
            if self.is_inside_main_circle(pos.x, pos.y):
                if self.is_inside_number(pos.x, pos.y):
                    # Cycle through figure colors more frequently
                    color = self.figure_colors[self.current_fg_color_index]
                    self.current_fg_color_index = (self.current_fg_color_index + 2) % len(self.figure_colors)
                else:
                    # Cycle through background colors more frequently
                    color = self.background_colors[self.current_bg_color_index]
                    self.current_bg_color_index = (self.current_bg_color_index + 2) % len(self.background_colors)
                
                self.draw_circle_with_gradient(circles_draw, pos, radius, color)


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

