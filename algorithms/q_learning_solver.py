# algorithms/q_learning_solver.py

import pygame
import numpy as np
from maze.generator import generate_maze
from maze.renderer import draw_maze

CELL_SIZE = 10
MAZE_ROWS, MAZE_COLS = 41, 41
WINDOW_WIDTH = MAZE_COLS * CELL_SIZE
WINDOW_HEIGHT = MAZE_ROWS * CELL_SIZE
FPS = 60

class QLearningAgent:
    def __init__(self, shape, alpha=0.1, gamma=0.9, epsilon=1.0):
        self.q_table = np.zeros(shape + (4,))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)
        return np.argmax(self.q_table[state[0], state[1]])

    def learn(self, s, a, r, ns):
        max_q = np.max(self.q_table[ns[0], ns[1]])
        self.q_table[s[0], s[1], a] += self.alpha * (r + self.gamma * max_q - self.q_table[s[0], s[1], a])

def run_q_learning_solver():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Q-Learning Maze Solver")
    clock = pygame.time.Clock()

    maze = generate_maze(MAZE_ROWS, MAZE_COLS)
    start = (1, 1)
    goal = (MAZE_ROWS - 2, MAZE_COLS - 2)
    agent = QLearningAgent(maze.shape)

    episodes = 1000
    print("Training Q-learning agent...")
    for ep in range(episodes):
        state = start
        done = False
        agent.epsilon = max(0.05, agent.epsilon * 0.995)  # Decay exploration
        steps = 0
        while not done and steps < 1000:
            action = agent.choose_action(state)
            dx, dy = [(0,1), (0,-1), (1,0), (-1,0)][action]
            nx, ny = state[0] + dx, state[1] + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx][ny] == 0:
                next_state = (nx, ny)
            else:
                next_state = state
            reward = 10 if next_state == goal else -0.1
            done = next_state == goal
            agent.learn(state, action, reward, next_state)
            state = next_state
            steps += 1
    print("Training completed.")

    # Test phase with animation
    print("Testing learned policy...")
    agent.epsilon = 0.0  # No more exploration
    state = start
    trail = [state]
    visited = set()
    steps = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if state == goal or steps > 2000:
            running = False
            continue

        action = agent.choose_action(state)
        dx, dy = [(0,1), (0,-1), (1,0), (-1,0)][action]
        nx, ny = state[0] + dx, state[1] + dy
        next_state = (nx, ny)

        if (0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1]
                and maze[nx][ny] == 0 and next_state not in visited):
            state = next_state
            trail.append(state)
            visited.add(state)
        else:
            # Try another action if stuck
            state = state  # stay in place

        screen.fill((0, 0, 0))
        draw_maze(screen, maze, state, goal, trail, CELL_SIZE)
        pygame.display.flip()
        clock.tick(FPS)
        steps += 1

    print("Maze traversal ended. Press close to exit.")
    # Keep window open until user closes it
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
