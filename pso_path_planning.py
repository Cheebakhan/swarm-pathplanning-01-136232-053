import numpy as np
import matplotlib.pyplot as plt
import random

# ==========================================
# CONFIGURATION & REPRODUCIBILITY
# ==========================================
ROLL_NUMBER_SEED = 136232053  # Seed derived from roll number 01-136232-053
GRID_SIZE = 20
OBSTACLE_DENSITY = 0.22       # 22% blocked cells
NUM_WAYPOINTS = 5             # Intermediate waypoints optimized by PSO
NUM_PARTICLES = 60            # Swarm population size
MAX_ITER = 150                # Convergence iterations

# Hyperparameters
W_MAX = 0.9                   # Initial inertia weight
W_MIN = 0.4                   # Final inertia weight
C1 = 1.8                      # Cognitive acceleration
C2 = 1.8                      # Social acceleration
COLLISION_PENALTY = 500.0     # Penalty for obstacle / boundary violations

# ==========================================
# 1. ENVIRONMENT GENERATION
# ==========================================
def generate_environment(grid_size, density, seed):
    random.seed(seed)
    np.random.seed(seed)

    grid = np.zeros((grid_size, grid_size), dtype=int)
    num_obstacles = int(grid_size * grid_size * density)

    all_cells = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    obstacle_indices = set(random.sample(range(len(all_cells)), num_obstacles))

    for idx in obstacle_indices:
        r, c = all_cells[idx]
        grid[r, c] = 1

    free_cells = [cell for idx, cell in enumerate(all_cells) if idx not in obstacle_indices]
    if len(free_cells) < 2:
        raise ValueError("Grid density too high; cannot find free start and goal.")

    start, goal = random.sample(free_cells, 2)
    return grid, np.array(start, dtype=float), np.array(goal, dtype=float)

# ==========================================
# 2. PATH DISCRETIZATION & COST FUNCTION
# ==========================================
def get_line_cells(p1, p2):
    """Interpolate continuous coordinates into discrete grid cells."""
    dist = np.linalg.norm(p2 - p1)
    num_steps = max(int(np.ceil(dist * 3)), 2)
    t = np.linspace(0, 1, num_steps)
    interp = (1 - t)[:, None] * p1 + t[:, None] * p2
    return np.round(interp).astype(int)

def evaluate_path(particle, start, goal, grid, grid_size):
    """Calculates total path length plus collision/boundary penalties."""
    waypoints = particle.reshape(-1, 2)
    full_path = np.vstack([start, waypoints, goal])

    total_length = 0.0
    penalty = 0.0

    for i in range(len(full_path) - 1):
        p1, p2 = full_path[i], full_path[i + 1]
        total_length += np.linalg.norm(p2 - p1)

        cells = get_line_cells(p1, p2)
        for r, c in cells:
            if r < 0 or r >= grid_size or c < 0 or c >= grid_size:
                penalty += COLLISION_PENALTY
            elif grid[r, c] == 1:
                penalty += COLLISION_PENALTY

    return total_length + penalty

# ==========================================
# 3. PARTICLE SWARM OPTIMIZATION (PSO)
# ==========================================
def run_pso(grid, start, goal, grid_size, num_waypoints=NUM_WAYPOINTS,
            num_particles=NUM_PARTICLES, max_iter=MAX_ITER):
    dim = num_waypoints * 2

    # Initialize positions along the start-to-goal vector with bounded noise
    positions = np.zeros((num_particles, dim))
    t = np.linspace(0.1, 0.9, num_waypoints)
    base_waypoints = (1 - t)[:, None] * start + t[:, None] * goal

    for i in range(num_particles):
        noise = np.random.uniform(-grid_size * 0.25, grid_size * 0.25, (num_waypoints, 2))
        init_pos = np.clip(base_waypoints + noise, 0, grid_size - 1)
        positions[i] = init_pos.flatten()

    velocities = np.random.uniform(-1.0, 1.0, (num_particles, dim))
    pbest_positions = np.copy(positions)
    pbest_scores = np.array([evaluate_path(p, start, goal, grid, grid_size) for p in positions])

    gbest_idx = np.argmin(pbest_scores)
    gbest_position = np.copy(pbest_positions[gbest_idx])
    gbest_score = pbest_scores[gbest_idx]

    v_max = grid_size * 0.15

    for it in range(max_iter):
        # Linearly decreasing inertia weight
        w = W_MAX - (W_MAX - W_MIN) * (it / max_iter)

        for i in range(num_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            velocities[i] = (w * velocities[i]
                             + C1 * r1 * (pbest_positions[i] - positions[i])
                             + C2 * r2 * (gbest_position - positions[i]))
            
            # Velocity clamping and position updates
            velocities[i] = np.clip(velocities[i], -v_max, v_max)
            positions[i] = np.clip(positions[i] + velocities[i], 0, grid_size - 1)

            # Evaluate score
            score = evaluate_path(positions[i], start, goal, grid, grid_size)
            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest_positions[i] = np.copy(positions[i])

                if score < gbest_score:
                    gbest_score = score
                    gbest_position = np.copy(positions[i])

        if (it + 1) % 25 == 0 or it == 0:
            print(f"Iteration {it+1:3d}/{max_iter} | Best Cost: {gbest_score:.2f}")

    return gbest_position, gbest_score

# ==========================================
# 4. VISUALIZATION
# ==========================================
def plot_results(grid, start, goal, best_particle, grid_size, seed, final_cost):
    waypoints = best_particle.reshape(-1, 2)
    path = np.vstack([start, waypoints, goal])

    # Check validity
    has_collision = False
    for i in range(len(path) - 1):
        cells = get_line_cells(path[i], path[i + 1])
        for r, c in cells:
            if r < 0 or r >= grid_size or c < 0 or c >= grid_size or grid[r, c] == 1:
                has_collision = True
                break

    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='binary', origin='upper', extent=[0, grid_size, grid_size, 0])

    # Path lines and waypoints (col = x, row = y)
    plt.plot(path[:, 1] + 0.5, path[:, 0] + 0.5, color='#007acc', linestyle='-', linewidth=2.5, label='PSO Path')
    plt.scatter(path[1:-1, 1] + 0.5, path[1:-1, 0] + 0.5, color='#007acc', marker='o', s=40, label='Waypoints')

    # Start and Goal markers
    plt.scatter(start[1] + 0.5, start[0] + 0.5, color='#28a745', s=160, marker='s', edgecolors='black', label=f'Start ({int(start[0])}, {int(start[1])})')
    plt.scatter(goal[1] + 0.5, goal[0] + 0.5, color='#dc3545', s=160, marker='*', edgecolors='black', label=f'Goal ({int(goal[0])}, {int(goal[1])})')

    plt.title(f"PSO Path Planning (Seed: {seed})\nStatus: {'Collision-Free' if not has_collision else 'Collision Detected'} | Best Cost: {final_cost:.2f}")
    plt.xlabel("Grid Column (X)")
    plt.ylabel("Grid Row (Y)")
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("pso_path_result.png", dpi=300)
    plt.show()

# ==========================================
# 5. EXECUTION ENTRY POINT
# ==========================================
if __name__ == "__main__":
    print(f"Generating problem with Roll Number Seed: {ROLL_NUMBER_SEED}")
    grid, start, goal = generate_environment(GRID_SIZE, OBSTACLE_DENSITY, ROLL_NUMBER_SEED)
    print(f"Grid Size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Start Point: {start.astype(int)}, Goal Point: {goal.astype(int)}")

    best_particle, final_cost = run_pso(grid, start, goal, GRID_SIZE)
    print(f"\nOptimization Finished. Final Cost: {final_cost:.2f}")

    plot_results(grid, start, goal, best_particle, GRID_SIZE, ROLL_NUMBER_SEED, final_cost)