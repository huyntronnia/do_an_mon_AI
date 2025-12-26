import pygame
import random
import os
from collections import namedtuple
from enum import Enum

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# --- CẤU HÌNH ---
BLOCK_SIZE = 20
SPEED = 200 #

# --- MÀU SẮC ---
BLACK = (0, 0, 0)
GRID_COLOR = (35, 35, 35)
WHITE = (255, 255, 255)
RED = (200, 50, 50)
SNAKE_COLOR = (30, 144, 255)    
SNAKE_OUTLINE = (0, 0, 205)
EYE_COLOR = (0, 0, 0)

# Class quản lý hạt nổ (Particle)
class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-3, 3)
        self.radius = random.randint(2, 5)
        self.life = 20 # Sống trong 20 frames

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        if self.radius > 0:
            self.radius -= 0.1 # Nhỏ dần

    def draw(self, surface):
        if self.life > 0 and self.radius > 0:
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.radius))

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        pygame.init()
        pygame.font.init()
        
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h), pygame.DOUBLEBUF)
        pygame.display.set_caption('Snake AI - Visualization')
        self.clock = pygame.time.Clock()
        
        # Load ảnh
        self.use_image = False
        img_path = "/Users/kimquyen/Documents/SIU - TKVM/do_an_mon_AI/Quyen/picture/apple.png" 
        try:
            if os.path.exists(img_path):
                raw_img = pygame.image.load(img_path).convert_alpha()
                self.apple_img = pygame.transform.smoothscale(raw_img, (BLOCK_SIZE + 4, BLOCK_SIZE + 4))
                self.use_image = True
        except Exception as e:
            print(f"Lỗi ảnh: {e}")

        self.particles = [] # List chứa các hạt nổ
        self.current_path = [] # List chứa đường đi AI dự tính
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.particles.clear()
        self.current_path = []

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    # Hàm nhận path từ AI để vẽ
    def set_path(self, path):
        self.current_path = path

    def _spawn_particles(self, x, y):
        # Tạo 15 hạt tại vị trí ăn mồi
        for _ in range(15):
            self.particles.append(Particle(x + BLOCK_SIZE//2, y + BLOCK_SIZE//2, RED))

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        self._move(action)
        self.snake.insert(0, self.head)
        
        game_over = False
        if self.is_collision():
            game_over = True
            return game_over, self.score

        if self.head == self.food:
            self.score += 1
            print(f"Score: {self.score}")
            # Tạo hiệu ứng nổ
            self._spawn_particles(self.head.x, self.head.y)
            self._place_food()
        else:
            self.snake.pop()
        
        self._update_ui()
        self.clock.tick(SPEED)
        return game_over, self.score

    def is_collision(self, pt=None):
        if pt is None: pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _draw_grid(self):
        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(self.display, GRID_COLOR, (x, 0), (x, self.h))
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.display, GRID_COLOR, (0, y), (self.w, y))

    def _draw_snake_body(self, color, inflate_size):
        for i, pt in enumerate(self.snake):
            center = (pt.x + BLOCK_SIZE//2, pt.y + BLOCK_SIZE//2)
            radius = (BLOCK_SIZE//2) + inflate_size
            pygame.draw.circle(self.display, color, center, radius)
            if i > 0:
                prev_pt = self.snake[i-1]
                x1 = min(pt.x, prev_pt.x) + BLOCK_SIZE//2
                y1 = min(pt.y, prev_pt.y) + BLOCK_SIZE//2
                x2 = max(pt.x, prev_pt.x) + BLOCK_SIZE//2
                y2 = max(pt.y, prev_pt.y) + BLOCK_SIZE//2
                if x1 == x2:
                    rect = pygame.Rect(x1 - radius, y1, radius * 2, y2 - y1)
                else:
                    rect = pygame.Rect(x1, y1 - radius, x2 - x1, radius * 2)
                pygame.draw.rect(self.display, color, rect)

    def _draw_eyes(self, head_rect):
        center_x, center_y = head_rect.center
        eye_radius = 3; pupil_radius = 1.5; off_front = 6; off_side = 5
        eye1, eye2 = (0,0), (0,0); pupil1, pupil2 = (0,0), (0,0); p_off = 1
        
        if self.direction == Direction.RIGHT:
            eye1 = (center_x + off_front, center_y - off_side); eye2 = (center_x + off_front, center_y + off_side)
            pupil1 = (eye1[0] + p_off, eye1[1]); pupil2 = (eye2[0] + p_off, eye2[1])
        elif self.direction == Direction.LEFT:
            eye1 = (center_x - off_front, center_y - off_side); eye2 = (center_x - off_front, center_y + off_side)
            pupil1 = (eye1[0] - p_off, eye1[1]); pupil2 = (eye2[0] - p_off, eye2[1])
        elif self.direction == Direction.UP:
            eye1 = (center_x - off_side, center_y - off_front); eye2 = (center_x + off_side, center_y - off_front)
            pupil1 = (eye1[0], eye1[1] - p_off); pupil2 = (eye2[0], eye2[1] - p_off)
        elif self.direction == Direction.DOWN:
            eye1 = (center_x - off_side, center_y + off_front); eye2 = (center_x + off_side, center_y + off_front)
            pupil1 = (eye1[0], eye1[1] + p_off); pupil2 = (eye2[0], eye2[1] + p_off)
            
        pygame.draw.circle(self.display, WHITE, eye1, eye_radius)
        pygame.draw.circle(self.display, WHITE, eye2, eye_radius)
        pygame.draw.circle(self.display, BLACK, pupil1, pupil_radius)
        pygame.draw.circle(self.display, BLACK, pupil2, pupil_radius)

    def _update_ui(self):
        self.display.fill(BLACK)

        # Vẽ Rắn
        self._draw_snake_body(SNAKE_OUTLINE, 2)
        self._draw_snake_body(SNAKE_COLOR, 0)
        head_rect = pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE, BLOCK_SIZE)
        self._draw_eyes(head_rect)

        # Vẽ Thức ăn
        off_x = (BLOCK_SIZE - self.apple_img.get_width()) // 2
        off_y = (BLOCK_SIZE - self.apple_img.get_height()) // 2
        if self.use_image:
            self.display.blit(self.apple_img, (self.food.x + off_x, self.food.y + off_y))
        else:
            pygame.draw.circle(self.display, RED, (self.food.x + BLOCK_SIZE//2, self.food.y + BLOCK_SIZE//2), BLOCK_SIZE//2 - 2)

        # Vẽ Particles (Hiệu ứng)
        for p in self.particles:
            p.update()
            p.draw(self.display)
        # Xóa hạt đã chết
        self.particles = [p for p in self.particles if p.life > 0]

        # Vẽ Điểm
        font = pygame.font.SysFont('arial', 30, bold=True)
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [10, 10])
        
        pygame.display.flip()

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT: x += BLOCK_SIZE
        elif direction == Direction.LEFT: x -= BLOCK_SIZE
        elif direction == Direction.DOWN: y += BLOCK_SIZE
        elif direction == Direction.UP: y -= BLOCK_SIZE
        self.head = Point(x, y)
        self.direction = direction