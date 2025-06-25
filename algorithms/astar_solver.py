# algorithms/astar_solver.py

import pygame
import heapq
from maze.generator import generate_maze
from maze.renderer import draw_maze

CELL_SIZE = 10
MAZE_ROWS, MAZE_COLS = 51, 51
WINDOW_WIDTH = MAZE_COLS * CELL_SIZE
WINDOW_HEIGHT = MAZE_ROWS * CELL_SIZE
FPS = 30

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(pos, maze):
    neighbors = []
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx][ny] == 0:
            neighbors.append((nx, ny))
    return neighbors

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def run_astar_solver():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("A* Maze Solver")
    clock = pygame.time.Clock()

    maze = generate_maze(MAZE_ROWS, MAZE_COLS)
    start = (1, 1)
    goal = (MAZE_ROWS - 2, MAZE_COLS - 2)

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            break
        for neighbor in get_neighbors(current, maze):
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                if neighbor not in [i[1] for i in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        screen.fill((0,0,0))
        trail = reconstruct_path(came_from, current)
        draw_maze(screen, maze, current, goal, trail, CELL_SIZE)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
