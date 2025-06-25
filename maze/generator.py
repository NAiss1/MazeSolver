import numpy as np
import random

def generate_maze(rows, cols):
    maze = np.ones((rows, cols), dtype=int)
    def carve(x, y):
        dirs = [(0,2), (0,-2), (2,0), (-2,0)]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 1 <= nx < rows-1 and 1 <= ny < cols-1 and maze[nx][ny] == 1:
                maze[nx][ny] = 0
                maze[x + dx//2][y + dy//2] = 0
                carve(nx, ny)
    maze[1][1] = 0
    carve(1, 1)
    return maze
