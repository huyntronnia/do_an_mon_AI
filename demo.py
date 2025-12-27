import torch
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet
import numpy as np
import os
from agent import Agent 

def run_demo():
    model = Linear_QNet(14, 256, 3)
    
    model_path = 'D:\\snake\\do_an_mon_AI\\model\\model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("Đã load model thành công!")
    else:
        print("Không tìm thấy file model!")
        return

    game = SnakeGameAI()
    agent = Agent() 

    while True:
        state = agent.get_state(game)
        
        state0 = torch.tensor(state, dtype=torch.float)
        
        prediction = model(state0)
        move = torch.argmax(prediction).item()
        final_move = [0, 0, 0]
        final_move[move] = 1

        reward, done, score = game.play_step(final_move)

        if done:
            print(f'Game Over. Score: {score}')
            game.reset()

if __name__ == '__main__':
    run_demo()