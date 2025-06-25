# main.py

import argparse
from algorithms.genetic_solver import run_genetic_solver
from algorithms.q_learning_solver import run_q_learning_solver
from algorithms.astar_solver import run_astar_solver

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AI Maze Solver")
    parser.add_argument('--method', type=str, default='genetic', choices=['genetic', 'qlearning', 'astar'],
                        help='Choose the solving algorithm')
    args = parser.parse_args()

    if args.method == 'genetic':
        run_genetic_solver()
    elif args.method == 'qlearning':
        run_q_learning_solver()
    elif args.method == 'astar':
        run_astar_solver()
    else:
        print("Invalid method. Choose from: genetic, qlearning, astar.")
