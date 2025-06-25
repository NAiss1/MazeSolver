# maze/renderer.py
def draw_maze(screen, maze, agent_pos, goal, trail, cell_size):
    import pygame
    WHITE = (240, 240, 240)
    BLACK = (30, 30, 30)
    GREEN = (0, 255, 0)
    RED = (255, 60, 60)
    BLUE = (60, 150, 255)
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            color = WHITE if maze[i][j] == 0 else BLACK
            pygame.draw.rect(screen, color, (j*cell_size, i*cell_size, cell_size, cell_size))
    for t in trail:
        pygame.draw.rect(screen, BLUE, (t[1]*cell_size, t[0]*cell_size, cell_size, cell_size))
    gx, gy = goal
    pygame.draw.rect(screen, GREEN, (gy*cell_size, gx*cell_size, cell_size, cell_size))
    x, y = agent_pos
    pygame.draw.circle(screen, RED, (y*cell_size + cell_size//2, x*cell_size + cell_size//2), cell_size//3)