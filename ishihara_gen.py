
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

        self.create_boundary()


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

    def setup_number_transform(self):
        """Pre-compute coordinate transformation constants"""
        self.number_width = self.main_circle_radius * 1.4
        self.number_height = self.main_circle_radius * 1.4
        self.number_x = self.center_x - self.number_width/2
        self.number_y = self.center_y - self.number_height/2 - self.main_circle_radius * 0.1

    def create_boundary(self):
        """Modified boundary with better containment"""
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.space.add(body)

        rect_left = self.center_x - self.rect_width//2
        rect_right = self.center_x + self.rect_width//2
        rect_top = self.center_y - self.rect_height//2
        rect_bottom = self.center_y + self.rect_height//2

        # Add more segments for better containment
        segments = [
            [(rect_left, rect_bottom), (rect_right, rect_bottom)],  # bottom
            [(rect_left, rect_top), (rect_left, rect_bottom)],      # left
            [(rect_right, rect_top), (rect_right, rect_bottom)],    # right
            [(rect_left, rect_top), (rect_right, rect_top)],        # top
        ]

        for points in segments:
            segment = pymunk.Segment(body, points[0], points[1], 1.0)
            segment.friction = 0.9  # Increased friction
            segment.elasticity = 0.1
            self.space.add(segment)

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

    def find_best_circle_size(self, x, y, available_radii):
        """Find the largest circle that fits at the given position with no overlaps"""
        # Check against existing circles in the space
        existing_circles = [s for s in self.space.shapes if isinstance(s, pymunk.Circle)]
        
        # Calculate minimum required gap between circles (5% of radius)
        min_gap = 2  
        
        for radius in available_radii:
            fits = True
            
            # Check if circle would be too close to the edge of main circle
            dist_to_center = math.sqrt((x - self.center_x)**2 + (y - self.center_y)**2)
            if dist_to_center + radius + min_gap > self.main_circle_radius:
                continue
                
            # Check against all existing circles
            for shape in existing_circles:
                other_pos = shape.body.position
                min_dist = radius + shape.radius + min_gap
                actual_dist = math.sqrt((x - other_pos.x)**2 + (y - other_pos.y)**2)
                
                if actual_dist < min_dist:
                    fits = False
                    break
            
            if fits:
                return radius
        
        return None

    def add_circles_batch(self, num_circles):
        """Add circles using spiral distribution and adaptive sizing"""
        circles = []
        theta = math.pi * (3 - math.sqrt(5))  # Golden angle
        
        # Sort radii from largest to smallest
        available_radii = sorted(self.small_circle_radii, reverse=True)
        
        # Track attempted positions to avoid retrying same spots
        attempted_positions = set()
        
        attempts = 0
        max_attempts = num_circles * 3  # Allow some retries
        successful_placements = 0
        
        while successful_placements < num_circles and attempts < max_attempts:
            # Generate position using golden angle spiral
            i = attempts
            r = math.sqrt(i / num_circles) * self.main_circle_radius * 0.8
            theta_i = i * theta
            
            x = self.center_x + r * math.cos(theta_i)
            y = self.center_y + r * math.sin(theta_i)
            
            # Round position to grid to avoid nearly identical attempts
            grid_x = round(x / 10) * 10
            grid_y = round(y / 10) * 10
            pos_key = (grid_x, grid_y)
            
            if pos_key not in attempted_positions:
                attempted_positions.add(pos_key)
                
                # Find largest circle that fits at this position
                chosen_radius = self.find_best_circle_size(x, y, available_radii)
                
                if chosen_radius:
                    # Add some randomness to final position
                    jitter = chosen_radius * 0.1
                    x += random.uniform(-jitter, jitter)
                    y += random.uniform(-jitter, jitter)
                    
                    # Create physics body and shape
                    mass = 1.0
                    moment = pymunk.moment_for_circle(mass, 0, chosen_radius)
                    body = pymunk.Body(mass, moment)
                    body.position = x, y
                    
                    shape = pymunk.Circle(body, chosen_radius)
                    shape.friction = 0.9
                    shape.elasticity = 0.1
                    shape.collision_type = 1
                    
                    self.space.add(body, shape)
                    circles.append((shape, chosen_radius))
                    successful_placements += 1
            
            attempts += 1
        
        return circles


    def find_valid_position(self, radius, existing_circles, max_attempts=50):
        """Find a valid position for a new circle that's tangent to existing circles"""
        if not existing_circles:
            # First circle - place near center
            return self.center_x, self.center_y
            
        for _ in range(max_attempts):
            # Pick random existing circles to be tangent to
            if len(existing_circles) >= 2:
                c1, c2 = random.sample(existing_circles, 2)
                x1, y1 = c1[0]
                r1 = c1[1]
                x2, y2 = c2[0]
                r2 = c2[1]
                
                # Calculate positions where a circle of given radius would be tangent to both circles
                d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Skip if circles are too far apart or too close
                if d > r1 + r2 + 2 * radius + 2 or d < abs(r1 - r2):
                    continue
                    
                # Calculate possible positions
                pos_list = []
                
                # Base position
                base_x = x1 + (x2 - x1) * (r1 + radius) / d
                base_y = y1 + (y2 - y1) * (r1 + radius) / d
                
                # Perpendicular offset
                h = math.sqrt((r1 + radius)**2 - ((r1 + radius)**2 + d**2 - (r2 + radius)**2)**2 / (4 * d**2))
                dx = -h * (y2 - y1) / d
                dy = h * (x2 - x1) / d
                
                pos_list.append((base_x + dx, base_y + dy))
                pos_list.append((base_x - dx, base_y - dy))
                
                # Check each possible position
                for x, y in pos_list:
                    valid = True
                    
                    # Check if inside main circle
                    if math.sqrt((x - self.center_x)**2 + (y - self.center_y)**2) > self.main_circle_radius - radius:
                        continue
                        
                    # Check against all other existing circles
                    for other_x, other_y, other_r in [(c[0][0], c[0][1], c[1]) for c in existing_circles]:
                        min_dist = radius + other_r + 1  # 1 pixel gap
                        actual_dist = math.sqrt((x - other_x)**2 + (y - other_y)**2)
                        if actual_dist < min_dist:
                            valid = False
                            break
                    
                    if valid:
                        return x, y
            
            else:
                # Place adjacent to single existing circle
                c1 = existing_circles[0]
                x1, y1 = c1[0]
                r1 = c1[1]
                
                angle = random.uniform(0, 2 * math.pi)
                x = x1 + (r1 + radius + 1) * math.cos(angle)
                y = y1 + (r1 + radius + 1) * math.sin(angle)
                
                if math.sqrt((x - self.center_x)**2 + (y - self.center_y)**2) <= self.main_circle_radius - radius:
                    return x, y
        
        return None
    
    def find_largest_fitting_circle(self, x, y, available_radii, placed_circles):
        """Find the largest circle that fits at a given position"""
        spacing = 1  # Minimum spacing between circles
        
        for radius in available_radii:
            if not self.is_inside_main_circle(x, y):
                continue
                
            fits = True
            for (cx, cy), cr in placed_circles:
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < (radius + cr + spacing):
                    fits = False
                    break
                    
            if fits:
                return radius
        
        return None

    def fill_region(self, region_checker, grid, placed_circles, sizes_to_use):
        """Fill a region using maximum circle size diversity"""
        new_circles = []
        attempted_positions = set()
        
        # Initial placement of larger circles
        for radius in sizes_to_use[:3]:  # Start with largest sizes
            candidates = self.get_candidate_positions(placed_circles + new_circles, radius)
            random.shuffle(candidates)
            
            for x, y in candidates:
                pos_key = (round(x/5), round(y/5))  # Round to reduce position duplicates
                if pos_key in attempted_positions:
                    continue
                    
                attempted_positions.add(pos_key)
                if region_checker(x, y):
                    if self.check_position(x, y, radius, grid, placed_circles + new_circles):
                        new_circles.append(((x, y), radius))
        
        # Fill remaining spaces with smaller circles
        max_circles = 1000  # Increased limit for better coverage
        while len(new_circles) < max_circles:
            # Get candidates around existing circles
            all_current = placed_circles + new_circles
            candidates = self.get_candidate_positions(all_current, min(sizes_to_use))
            
            # Try each candidate position
            success = False
            for x, y in candidates:
                pos_key = (round(x/5), round(y/5))
                if pos_key in attempted_positions:
                    continue
                    
                attempted_positions.add(pos_key)
                if not region_checker(x, y):
                    continue
                    
                # Find largest circle that fits here
                radius = self.find_largest_fitting_circle(x, y, sizes_to_use, all_current)
                if radius:
                    new_circles.append(((x, y), radius))
                    success = True
                    break
            
            if not success:
                # If we can't place any more circles, try with smaller sizes
                if len(sizes_to_use) > 1:
                    sizes_to_use = sizes_to_use[1:]
                else:
                    break
        
        return new_circles

    def get_circle_sizes(self):
        """Create 4-5 distinct circle sizes with good distribution"""
        # Define 5 distinct sizes
        sizes = [48, 40, 32, 24, 16]  # pixels diameter
        return [s//2 for s in sizes]  # Convert to radii

    def enforce_size_distribution(self, circles, target_count):
        """Check if we have a good distribution of circle sizes"""
        size_counts = {}
        for _, radius in circles:
            size_counts[radius] = size_counts.get(radius, 0) + 1
        
        # Get target count per size (roughly equal distribution)
        target_per_size = target_count / len(self.get_circle_sizes())
        
        # Check if any size is overused
        for radius, count in size_counts.items():
            if count > target_per_size * 1.3:  # Allow 30% variance
                return False
        return True


    def get_circle_sizes(self):
        """Create 5 distinct circle sizes with quotas"""
        # Define sizes in pixels (diameter)
        sizes = [48, 40, 32, 24, 16]
        return [s//2 for s in sizes]  # Convert to radii

    def get_size_quota(self, current_circles, target_total, radius):
        """Check if we can still use this size based on quotas"""
        size_counts = {}
        for _, r in current_circles:
            size_counts[r] = size_counts.get(r, 0) + 1
        
        # Calculate target per size (20% each for 5 sizes)
        target_per_size = target_total / 5
        
        # Allow slight variance (10%)
        max_allowed = int(target_per_size * 1.1)
        current_count = size_counts.get(radius, 0)
        
        return current_count < max_allowed

    def select_next_radius(self, current_circles, target_total, available_positions):
        """Select next radius based on maintaining even distribution"""
        radii = self.get_circle_sizes()
        
        # Count current sizes
        size_counts = {r: 0 for r in radii}
        for _, r in current_circles:
            size_counts[r] += 1
        
        # Find most underrepresented size that fits
        target_per_size = target_total / 5
        min_count = float('inf')
        selected_radius = None
        
        # Shuffle radii to avoid always checking in same order
        shuffled_radii = radii.copy()
        random.shuffle(shuffled_radii)
        
        for radius in shuffled_radii:
            count = size_counts[radius]
            if count < min_count and count < target_per_size:
                # Verify this size fits in at least one available position
                for x, y in available_positions:
                    if self.check_position(x, y, radius, current_circles):
                        min_count = count
                        selected_radius = radius
        
        return selected_radius or random.choice(radii)

    def get_candidate_positions(self, region_checker, current_circles, spacing=2):
        """Get available positions for new circles"""
        positions = []
        
        if not current_circles:
            # Initial positions for empty region
            center_x = self.number_x + self.number_width/2
            center_y = self.number_y + self.number_height/2
            for angle in range(0, 360, 45):
                rad = math.radians(angle)
                x = center_x + math.cos(rad) * 30
                y = center_y + math.sin(rad) * 30
                if region_checker(x, y):
                    positions.append((x, y))
            return positions
        
        # Generate positions around existing circles
        for (cx, cy), cr in current_circles:
            # More positions around larger circles
            num_positions = int(12 * (cr / max(self.get_circle_sizes())))
            for i in range(num_positions):
                angle = (i / num_positions) * 2 * math.pi
                for dist_factor in [1.0, 1.2, 1.4]:
                    x = cx + math.cos(angle) * (cr + 20) * dist_factor
                    y = cy + math.sin(angle) * (cr + 20) * dist_factor
                    if region_checker(x, y):
                        positions.append((x, y))
        
        return positions

        """Check if a circle fits at position"""
        for (cx, cy), cr in current_circles:
            dist = math.sqrt((x - cx)**2 + (y - cy)**2)
            if dist < (radius + cr + spacing):
                return False
        return True

    def pack_number_region(self):
        """Pack number region with strictly enforced size distribution"""
        number_circles = []
        target_circles = 95  # Target for number region
        max_attempts = 15000
        attempts = 0
        
        while attempts < max_attempts and len(number_circles) < target_circles:
            # Get candidate positions
            positions = self.get_candidate_positions(
                self.is_inside_number,
                number_circles
            )
            
            if not positions:
                break
                
            # Select radius based on maintaining distribution
            radius = self.select_next_radius(number_circles, target_circles, positions)
            
            # Try positions with selected radius
            random.shuffle(positions)
            placed = False
            
            for x, y in positions:
                if self.check_position(x, y, radius, number_circles):
                    number_circles.append(((x, y), radius))
                    placed = True
                    break
            
            if not placed:
                attempts += 1
        
        return number_circles

    def pack_background(self, number_circles):
        """Pack background with strictly enforced size distribution"""
        background_circles = []
        target_circles = 400  # Target for background
        max_attempts = 25000
        attempts = 0
        
        def is_valid_background(x, y):
            return (self.is_inside_main_circle(x, y) and 
                    not self.is_inside_number(x, y))
        
        while attempts < max_attempts and len(background_circles) < target_circles:
            # Get candidate positions
            positions = self.get_candidate_positions(
                is_valid_background,
                background_circles + number_circles
            )
            
            if not positions:
                break
                
            # Select radius based on maintaining distribution
            radius = self.select_next_radius(background_circles, target_circles, positions)
            
            # Try positions with selected radius
            random.shuffle(positions)
            placed = False
            
            for x, y in positions:
                if self.check_position(x, y, radius, background_circles + number_circles):
                    background_circles.append(((x, y), radius))
                    placed = True
                    break
            
            if not placed:
                attempts += 1
        
        return background_circles

    def select_next_radius(self, current_circles, target_total, positions):
        """Select next radius with better termination conditions"""
        radii = self.get_circle_sizes()
        
        # Count current sizes
        size_counts = {r: 0 for r in radii}
        for _, r in current_circles:
            size_counts[r] += 1
        
        # Calculate target per size (20% each)
        target_per_size = target_total / len(radii)
        
        # Try each radius in random order
        shuffled_radii = radii.copy()
        random.shuffle(shuffled_radii)
        
        for radius in shuffled_radii:
            count = size_counts[radius]
            if count < target_per_size * 1.1:  # Allow 10% variance
                # Quick check if this radius might fit anywhere
                for x, y in random.sample(positions, min(len(positions), 10)):
                    if self.check_position(x, y, radius, current_circles):
                        return radius
        
        # If no radius satisfies distribution, return smallest that fits
        for radius in sorted(radii):
            for x, y in random.sample(positions, min(len(positions), 10)):
                if self.check_position(x, y, radius, current_circles):
                    return radius
        
        return min(radii)  # Fallback to smallest radius






    def check_position(self, x, y, radius, current_circles, number_circles=None, is_number_region=True, spacing=1):
        """Enhanced position checking with strict boundary enforcement"""
        # Check main circle boundary
        if not self.is_inside_main_circle(x, y):
            return False
            
        # Check region boundaries
        if is_number_region:
            if not self.is_inside_number(x, y):
                return False
        else:
            if self.is_inside_number(x, y):
                return False
                
            # For background circles, also check against number circles
            if number_circles:
                for (cx, cy), cr in number_circles:
                    dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                    if dist < (radius + cr + spacing):
                        return False
        
        # Check against current region's circles with tighter spacing
        for (cx, cy), cr in current_circles:
            dist = math.sqrt((x - cx)**2 + (y - cy)**2)
            if dist < (radius + cr + spacing):
                return False
                
        return True

    def pack_region(self, number_circles=None, is_number_region=True):
        """Improved region packing with tighter spacing"""
        circles = []
        attempted_positions = set()
        spacing = 1  # Reduced spacing
        
        # Set parameters based on region
        if is_number_region:
            target_circles = 95
            region_checker = self.is_inside_number
        else:
            target_circles = 500  # Increased for tighter packing
            region_checker = lambda x, y: (
                self.is_inside_main_circle(x, y) and 
                not self.is_inside_number(x, y)
            )
        
        # Initialize grid for position tracking with tighter grid
        grid_size = 8  # Reduced grid size for more potential positions
        grid_positions = []
        
        # Generate initial grid positions
        x_range = range(
            int(self.center_x - self.main_circle_radius),
            int(self.center_x + self.main_circle_radius),
            grid_size
        )
        y_range = range(
            int(self.center_y - self.main_circle_radius),
            int(self.center_y + self.main_circle_radius),
            grid_size
        )
        
        for y in y_range:
            for x in x_range:
                if region_checker(x, y):
                    # Add slight randomization to grid positions
                    jitter = grid_size * 0.25
                    x_jitter = x + random.uniform(-jitter, jitter)
                    y_jitter = y + random.uniform(-jitter, jitter)
                    grid_positions.append((x_jitter, y_jitter))
        
        random.shuffle(grid_positions)
        positions = grid_positions
        
        max_attempts = 2000  # Increased for better coverage
        attempts = 0
        
        while attempts < max_attempts and len(circles) < target_circles:
            if not positions:
                # Generate new positions around existing circles
                positions = []
                for (cx, cy), cr in circles:
                    for angle in range(0, 360, 30):  # More angles for tighter packing
                        rad = math.radians(angle)
                        for dist_factor in [1.0, 1.1, 1.2]:  # Tighter distance factors
                            new_x = cx + math.cos(rad) * (cr * 2 + spacing) * dist_factor
                            new_y = cy + math.sin(rad) * (cr * 2 + spacing) * dist_factor
                            if region_checker(new_x, new_y):
                                positions.append((new_x, new_y))
                
                if not positions:
                    break
                random.shuffle(positions)
            
            # Select radius based on current distribution
            radius = self.select_next_radius(circles, target_circles, positions)
            
            # Try each position
            success = False
            for x, y in positions[:]:
                pos_key = (round(x/grid_size), round(y/grid_size))
                if pos_key not in attempted_positions:
                    attempted_positions.add(pos_key)
                    
                    if self.check_position(x, y, radius, circles, number_circles, is_number_region, spacing):
                        circles.append(((x, y), radius))
                        positions.remove((x, y))
                        success = True
                        
                        # Add new positions around this circle
                        for angle in range(0, 360, 30):
                            rad = math.radians(angle)
                            for dist_factor in [1.0, 1.1, 1.2]:
                                new_x = x + math.cos(rad) * (radius * 2 + spacing) * dist_factor
                                new_y = y + math.sin(rad) * (radius * 2 + spacing) * dist_factor
                                if region_checker(new_x, new_y):
                                    positions.append((new_x, new_y))
                        break
            
            if not success:
                attempts += 1
            
            # Shuffle remaining positions periodically
            if attempts % 50 == 0:
                random.shuffle(positions)
        
        return circles

    def pack_circles(self):
        """Pack circles with strict boundary enforcement"""
        # First pack the number region
        number_circles = self.pack_region(is_number_region=True)
        
        # Then pack the background with reference to number circles
        background_circles = self.pack_region(number_circles, is_number_region=False)
        
        # Convert to physics bodies
        circles = []
        for (x, y), radius in number_circles + background_circles:
            body = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0, radius))
            body.position = (x, y)
            shape = pymunk.Circle(body, radius)
            shape.friction = 0.9
            shape.elasticity = 0.1
            self.space.add(body, shape)
            circles.append((shape, radius))
        
        return circles

    def run_physics_simulation(self):
        """Replace physics simulation with fixed packing"""
        return self.pack_circles()
           

    def create_packing_grid(self):
        """Create a grid to track circle placement"""
        # Use smallest circle radius to determine grid size
        self.min_radius = min(self.small_circle_radii)
        self.grid_cell_size = self.min_radius * 2
        
        # Calculate grid dimensions
        grid_width = int(self.width / self.grid_cell_size) + 1
        grid_height = int(self.height / self.grid_cell_size) + 1
        
        # Initialize grid
        return [[None for _ in range(grid_width)] for _ in range(grid_height)]

    def get_grid_coords(self, x, y):
        """Convert world coordinates to grid coordinates"""
        return (int(x / self.grid_cell_size), int(y / self.grid_cell_size))

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







    # def run_physics_simulation(self):
    #     """Modified physics simulation with better initial placement"""
    #     circles = []
    #     batch_size = 30  # Smaller batch size for more controlled placement
    #     batches = 0
    #     max_batches = 40
        
    #     coverage_threshold = 0.85  # Target coverage percentage
        
    #     while batches < max_batches:
    #         new_circles = self.add_circles_batch(batch_size)
    #         if not new_circles:
    #             break
                
    #         circles.extend(new_circles)
            
    #         # More gentle physics simulation
    #         for _ in range(30):
    #             self.space.step(1/60.0)
                
    #             # Apply gentle force to settle circles
    #             for circle, _ in new_circles:
    #                 circle.body.apply_force_at_local_point((0, 200.0), (0, 0))
                    
    #                 # Add damping to reduce bouncing
    #                 circle.body.velocity *= 0.95
    #                 circle.body.angular_velocity *= 0.95
            
    #         batches += 1
            
    #         # Check coverage after each batch
    #         coverage = self.check_coverage(circles)
    #         if coverage > coverage_threshold:
    #             break
        
    #     # Final gentle settling
    #     for _ in range(60):
    #         self.space.step(1/60.0)
    #         for circle, _ in circles:
    #             circle.body.velocity *= 0.98
    #             circle.body.angular_velocity *= 0.98
        
    #     return circles




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

