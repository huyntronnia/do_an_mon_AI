import pygame
import random
from collections import namedtuple

# Định nghĩa hướng đi
class Direction:
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')
BLOCK_SIZE = 20
SPEED = 10  # Tốc độ game (để AI chạy nhanh thì tăng lên)

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # Init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Khởi tạo lại trạng thái game
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

    def play_step(self):
        # 1. Thu thập sự kiện người dùng
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            
            # --- Code thêm để chơi bằng bàn phím ---
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP and self.direction != Direction.DOWN:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
                    self.direction = Direction.DOWN
            # ---------------------------------------
        
        # 2. Di chuyển (theo hướng self.direction hiện tại)
        self._move(self.direction)
        self.snake.insert(0, self.head)
        
        # 3. Kiểm tra game over
        game_over = False
        if self.is_collision():
            game_over = True
            return game_over, self.score

        # 4. Ăn mồi hoặc di chuyển tiếp
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. Cập nhật giao diện
        self._update_ui()
        self.clock.tick(SPEED) # Tốc độ game (tăng giảm số SPEED ở đầu file để khó/dễ hơn)
        
        return game_over, self.score

    def is_collision(self, pt=None):
        if pt is None: pt = self.head
        # Đâm tường
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Đâm thân
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill((0,0,0)) # Màu đen
        for pt in self.snake:
            pygame.draw.rect(self.display, (0, 255, 0), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, (255, 0, 0), pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.flip()

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT: x += BLOCK_SIZE
        elif direction == Direction.LEFT: x -= BLOCK_SIZE
        elif direction == Direction.DOWN: y += BLOCK_SIZE
        elif direction == Direction.UP: y -= BLOCK_SIZE
        self.head = Point(x, y)
if __name__ == '__main__':
    game = SnakeGameAI()
    
    # Vòng lặp game
    while True:
        # Không cần action = ... nữa vì ta dùng bàn phím
        game_over, score = game.play_step()
        
        if game_over:
            break
            
    print('Final Score:', score)
    pygame.quit() # Thêm dòng này để tắt cửa sổ game sạch sẽ