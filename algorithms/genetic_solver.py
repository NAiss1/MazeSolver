
import pygame
import numpy as np
import random
from maze.generator import generate_maze
from maze.renderer import draw_maze

CELL_SIZE = 10
MAZE_ROWS, MAZE_COLS = 31, 31
WINDOW_WIDTH = MAZE_COLS * CELL_SIZE
WINDOW_HEIGHT = MAZE_ROWS * CELL_SIZE
POP_SIZE = 1000
GENOME_LENGTH = 800
MUTATION_RATE = 0.05
FPS = 60

WHITE = (240, 240, 240)
BLACK = (30, 30, 30)
GREEN = (0, 255, 0)
RED = (255, 60, 60)
BLUE = (60, 150, 255)
import imageio
frames = []
record = True  # Set to False to disable

def run_genetic_solver():
    
    while True:
        solved = run_genetic_generation_once()
        if solved:
            break
        else:
            print("Retrying from scratch...")
def random_genome():
    return [random.randint(0, 3) for _ in range(GENOME_LENGTH)]

def mutate(genome, generation=1, max_generations=500):
    dynamic_rate = MUTATION_RATE * (1 - generation / max_generations) + 0.05
    return [
        gene if random.random() > dynamic_rate else random.randint(0, 3)
        for gene in genome
    ]


def crossover(parent1, parent2):
    cut = random.randint(0, GENOME_LENGTH - 1)
    return parent1[:cut] + parent2[cut:]

def simulate_position(genome, maze, start):
    pos = list(start)
    for move in genome:
        dx, dy = [(0,1), (0,-1), (1,0), (-1,0)][move]
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx][ny] == 0:
            pos = [nx, ny]
    return tuple(pos)

def fitness(genome, maze, start, goal):
    pos = list(start)
    visited = set()
    repeat_penalty = 0
    reached_goal = False
    steps = 0

    for move in genome:
        dx, dy = [(0,1), (0,-1), (1,0), (-1,0)][move]
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx][ny] == 0:
            pos = [nx, ny]
            if tuple(pos) in visited:
                repeat_penalty += 1
            visited.add(tuple(pos))
            if tuple(pos) == goal:
                reached_goal = True
                break
        steps += 1

    dist = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    score = len(visited) * 3 - dist - repeat_penalty

    if reached_goal:
        score += 10000

    return score


def evolve(population, maze, start, goal):
    scored = [(fitness(g, maze, start, goal), g) for g in population]
    scored.sort(reverse=True)
    
    # Keep top 5%
    elites = [g for _, g in scored[:int(0.05 * POP_SIZE)]]
    
    # Breed rest from top 25%
    parents_pool = [g for _, g in scored[:int(0.25 * POP_SIZE)]]
    new_pop = elites.copy()
    
    while len(new_pop) < POP_SIZE:
        parent1, parent2 = random.sample(parents_pool, 2)
        child = crossover(parent1, parent2)
        child = mutate(child)
        new_pop.append(child)

    return new_pop, scored[0][1]


def simulate_genome(screen, genome, maze, start, goal):
    pos = list(start)
    trail = []
    clock = pygame.time.Clock()
    for move in genome:
        pygame.time.delay(10)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        dx, dy = [(0,1), (0,-1), (1,0), (-1,0)][move]
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx][ny] == 0:
            pos = [nx, ny]
            trail.append(tuple(pos))
        screen.fill(BLACK)
        draw_maze(screen, maze, pos, goal, trail, CELL_SIZE)
        if record:
            frame = pygame.surfarray.array3d(pygame.display.get_surface())
            frame = frame.transpose([1, 0, 2])  # Fix orientation
            frames.append(frame)

        pygame.display.flip()
        clock.tick(FPS)

def run_genetic_generation_once():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Genetic Maze Solver")

    maze = generate_maze(MAZE_ROWS, MAZE_COLS)
    start = (1,1)
    goal = (MAZE_ROWS-2, MAZE_COLS-2)
    population = [random_genome() for _ in range(POP_SIZE)]

    generation = 0
    max_generations = 500
    best = None
    font = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()

    while generation < max_generations:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        population, best = evolve(population, maze, start, goal)
        generation += 1
        best_fit = fitness(best, maze, start, goal)
        print(f"Generation {generation}, Best Fitness: {best_fit:.2f}")

        # Live animation of best genome
        pos = list(start)
        trail = [tuple(pos)]
        for move in best:
            dx, dy = [(0,1), (0,-1), (1,0), (-1,0)][move]
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx][ny] == 0:
                pos = [nx, ny]
                trail.append(tuple(pos))
            if tuple(pos) == goal:
                if record and frames:
                    imageio.mimsave("genetic_solver_run.gif", frames, fps=10)
                    print("Saved animation to genetic_solver_run.gif")

                break

        screen.fill(BLACK)
        draw_maze(screen, maze, pos, goal, trail, CELL_SIZE)

        # Display generation and fitness
        gen_text = font.render(f"Gen: {generation}", True, (255, 255, 255))
        fit_text = font.render(f"Fitness: {int(best_fit)}", True, (255, 255, 255))
        screen.blit(gen_text, (10, 10))
        screen.blit(fit_text, (10, 30))

        pygame.display.flip()
        clock.tick(30)  # Slower = more visible

        if tuple(pos) == goal:
            print(f"Solved in generation {generation}!")
            break

    # Keep window open after training
    print("Training complete. Close the window to exit.")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

