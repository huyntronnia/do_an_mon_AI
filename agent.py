import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY) 
        
        # Lưu ý: Input size là 14 (do dùng logic check Trap mới)
        self.model = Linear_QNet(14, 256, 3) 
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        # --- LOGIC LOAD MODEL CŨ ĐỂ TRAIN TIẾP ---
        model_path = './model/model.pth'
        if os.path.exists(model_path):
            print("--> Đã tìm thấy model cũ. Đang load để train tiếp...")
            try:
                self.model.load_state_dict(torch.load(model_path))
                self.model.train() # Chế độ train
                
                # MẸO: Nếu load model cũ, ta giả định nó đã học được kha khá.
                # Ta set n_games > 80 để epsilon <= 0.
                # Điều này giúp rắn KHÔNG đi random ngu ngốc lúc đầu nữa mà dùng não ngay.
                self.n_games = 81 
                print("--> Load thành công! Tắt chế độ Random đầu game.")
            except Exception as e:
                print(f"--> Lỗi khi load model: {e}")
                print("--> Sẽ tạo model mới và train từ đầu.")
                self.n_games = 0
        else:
            print("--> Không thấy model cũ. Train từ đầu.")

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Check bẫy (Hàm is_trap mới trong game.py)
        trap_l = game.is_trap(point_l)
        trap_r = game.is_trap(point_r)
        trap_u = game.is_trap(point_u)
        trap_d = game.is_trap(point_d)
        
        # Mapping bẫy theo hướng nhìn
        trap_straight = (dir_r and trap_r) or (dir_l and trap_l) or (dir_u and trap_u) or (dir_d and trap_d)
        trap_right = (dir_u and trap_r) or (dir_d and trap_l) or (dir_l and trap_u) or (dir_r and trap_d)
        trap_left = (dir_d and trap_r) or (dir_u and trap_l) or (dir_r and trap_u) or (dir_l and trap_d)

        state = [
            # Danger
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food Location
            game.food.x < game.head.x, 
            game.food.x > game.head.x, 
            game.food.y < game.head.y, 
            game.food.y > game.head.y,
            
            # Trap
            trap_straight,
            trap_right,
            trap_left
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) 
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Epsilon-Greedy
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        
        # Nếu epsilon > 0 thì mới có tỷ lệ random
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    # Để record = 0 để nó luôn cố gắng lưu lại mốc mới từ phiên chạy này
    # Nếu muốn khắt khe hơn, bạn có thể set cứng record = 157
    record = 0 
    
    agent = Agent()
    game = SnakeGameAI()
    
    print("Training Started...")

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
                print(f"--> Đã lưu Model mới! Record: {record}")

            print(f'Game {agent.n_games}, Score {score}, Record {record}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            # Chỉ vẽ lại mỗi 10 game để tránh crash
            if agent.n_games % 10 == 0:
                try:
                    plot(plot_scores, plot_mean_scores)
                except:
                    pass

if __name__ == '__main__':
    train()