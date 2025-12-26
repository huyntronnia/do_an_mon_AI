import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
# Nếu không có file arial.ttf, pygame sẽ dùng font mặc định
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Màu sắc
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 60 # Tăng tốc độ game lên chút cho nhanh

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # Init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI - Flood Fill Version')
        self.clock = pygame.time.Clock()
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

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # Di chuyển
        self._move(action) 
        self.snake.insert(0, self.head)
        
        reward = 0
        game_over = False
        
        # Nếu chết hoặc đi lòng vòng quá lâu
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Nếu ăn mồi
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # Update UI
        self._update_ui()
        self.clock.tick(SPEED)
        
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Đâm tường
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Đâm thân
        if pt in self.snake[1:]:
            return True
        return False

    # --- HÀM MỚI: FLOOD FILL CHECK TRAP ---
# Thay thế hàm is_trap cũ trong game.py bằng hàm này
    def is_trap(self, point):
        """
        Kiểm tra xem từ điểm 'point' có đường về ĐUÔI rắn không?
        Nếu có đường về đuôi -> An toàn (False).
        Nếu không -> Bẫy/Ngõ cụt (True).
        """
        w_grid = self.w // BLOCK_SIZE
        h_grid = self.h // BLOCK_SIZE
        
        start_x = int(point.x // BLOCK_SIZE)
        start_y = int(point.y // BLOCK_SIZE)
        
        # 1. Check biên (Tường)
        if start_x < 0 or start_x >= w_grid or start_y < 0 or start_y >= h_grid:
            return True # Là bẫy

        # 2. Tạo lưới vật cản
        grid = np.zeros((w_grid, h_grid), dtype=int)
        for pt in self.snake:
            x_idx = int(pt.x // BLOCK_SIZE)
            y_idx = int(pt.y // BLOCK_SIZE)
            if 0 <= x_idx < w_grid and 0 <= y_idx < h_grid:
                grid[x_idx][y_idx] = 1 # 1 là vật cản
        
        # Đặc biệt: Cái đuôi hiện tại sẽ di chuyển khi đầu di chuyển
        # Nên vị trí đuôi hiện tại thực chất là ô TRỐNG (đích đến)
        tail = self.snake[-1]
        tail_x = int(tail.x // BLOCK_SIZE)
        tail_y = int(tail.y // BLOCK_SIZE)
        
        # Đánh dấu đuôi là đi được (trừ khi rắn mới ăn mồi, đuôi giữ nguyên, 
        # nhưng để đơn giản ta cứ coi đuôi là đích đến an toàn)
        grid[tail_x][tail_y] = 0 

        # Nếu điểm check trùng thân rắn (mà không phải đuôi) -> Bẫy
        if grid[start_x][start_y] == 1:
            return True
            
        # 3. BFS tìm đường về đuôi
        queue = [(start_x, start_y)]
        visited = set()
        visited.add((start_x, start_y))
        
        found_tail = False
        
        while queue:
            cx, cy = queue.pop(0)
            
            # Nếu chạm được đuôi -> An toàn
            if cx == tail_x and cy == tail_y:
                found_tail = True
                break

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w_grid and 0 <= ny < h_grid:
                    if grid[nx][ny] == 0 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        
        # Nếu tìm thấy đuôi thì KHÔNG PHẢI BẪY (False), ngược lại là True
        return not found_tail

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [Straight, Right, Left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # No change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # Right turn
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # Left turn

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)