#  Maze Solver with AI Algorithms (Genetic, Q-Learning, A*)

This project is an animated Python application that generates random mazes and solves them using AI algorithms:

-  **Genetic Algorithm** (evolution-inspired optimization)
-  **Q-Learning** (reinforcement learning)
-  **A\*** (shortest path search)

Each solver is visualized using **Pygame**, with animated traversal and real-time updates.

---

##  Features

-  Maze generation using **recursive backtracking**
-  Three solving strategies (`genetic`, `qlearning`, `astar`)
-  **Real-time animation** of AI learning
-  **Auto-restart** if genetic algorithm fails
-  Optional **GIF recording** of the run
-  Command-line interface to choose solver

---

## ⚙️ CLI Options

| Flag         | Description                           |
|--------------|---------------------------------------|
| `--method`   | `genetic`, `qlearning`, or `astar`    |

---

##  GIF Recording

- The genetic solver records frames and saves a **GIF** as `genetic_solver_run.gif` in the current folder.

---

##  Genetic Algorithm Notes

- Solves by evolving a population of random paths.
- If no solution is found in 500 generations, it restarts automatically.
- Fitness considers distance to goal, steps, and path quality.

---

##  Q-Learning Notes

- Learns state-action values (`Q-table`) over 1000 episodes.
- Uses epsilon-greedy exploration to gradually find an optimal path.

---

##  A* Notes

- Classical pathfinding using a priority queue and Manhattan distance.
- Fast and efficient for well-defined mazes.


---

##  License

MIT License
