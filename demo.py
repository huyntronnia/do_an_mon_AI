import torch
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet
import numpy as np
import os
from agent import Agent # Import class Agent để tái sử dụng hàm get_state

def run_demo():
    # 1. Khởi tạo lại kiến trúc mạng (INPUT SIZE PHẢI LÀ 14)
    model = Linear_QNet(14, 256, 3)
    
    # 2. Nạp trọng số từ file model.pth (Phiên bản 157 điểm)
    model_path = './model/model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("Đã load model 157 điểm!")
    else:
        print("Không tìm thấy file model!")
        return

    game = SnakeGameAI()
    agent = Agent() # Dùng agent để gọi hàm get_state có Flood Fill

    while True:
        # Lấy trạng thái (bao gồm cả Flood Fill check)
        state = agent.get_state(game)
        
        # Chuyển sang tensor
        state0 = torch.tensor(state, dtype=torch.float)
        
        # Dự đoán nước đi tối ưu
        prediction = model(state0)
        move = torch.argmax(prediction).item()
        final_move = [0, 0, 0]
        final_move[move] = 1

        # Thực hiện nước đi
        reward, done, score = game.play_step(final_move)

        if done:
            print(f'Game Over. Score: {score}')
            game.reset()

if __name__ == '__main__':
    run_demo()