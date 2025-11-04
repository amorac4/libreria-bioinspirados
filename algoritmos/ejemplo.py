import numpy as np
import matplotlib.pyplot as plt

# Objective function: Rosenbrock function
def rosenbrock(x, y, a=1, b=100):
    return (a - x) ** 2 + b * (y - x ** 2) ** 2

# Initialize parameters
num_cats = 10
max_iterations = 100
dim = 2  # Dimensions in the problem
mix_rate = 0.5  # Probability of cat being in seeking or tracing mode
seeking_memory_pool = 5  # Number of candidate points in seeking mode
seeking_range = 0.2  # Range of seeking mode
velocity_limit = 0.03  # Velocity limit in tracing mode

# Initialize cats randomly
cats = np.random.rand(num_cats, dim) * 2 - 1  # Initialize positions
velocities = np.zeros((num_cats, dim))  # Initialize velocities

# Function to update cats in seeking mode
def seeking_mode(cats, seeking_range, seeking_memory_pool):
    for i, cat in enumerate(cats):
        candidate_positions = cat + np.random.uniform(-1, 1, (seeking_memory_pool, dim)) * seeking_range
        candidate_fitness = np.array([rosenbrock(pos[0], pos[1]) for pos in candidate_positions])
        best_candidate_index = np.argmin(candidate_fitness)
        cats[i] = candidate_positions[best_candidate_index]

# Function to update cats in tracing mode
def tracing_mode(cats, velocities, velocity_limit):
    global_best = cats[np.argmin([rosenbrock(cat[0], cat[1]) for cat in cats])]
    for i, velocity in enumerate(velocities):
        velocities[i] += np.random.rand() * (global_best - cats[i])
        velocities[i] = np.clip(velocities[i], -velocity_limit, velocity_limit)
        cats[i] += velocities[i]

# Optimization loop
best_fitness_history = []

for iteration in range(max_iterations):
    for i, cat in enumerate(cats):
        if np.random.rand() < mix_rate:
            seeking_mode(cats[i:i+1], seeking_range, seeking_memory_pool)
        else:
            tracing_mode(cats[i:i+1], velocities[i:i+1], velocity_limit)
    
    # Record the best fitness
    best_fitness = np.min([rosenbrock(cat[0], cat[1]) for cat in cats])
    best_fitness_history.append(best_fitness)

# Plotting the optimization process
plt.figure(figsize=(10, 6))
plt.plot(best_fitness_history, label='Best Fitness over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Best Fitness')
plt.title('Cat Swarm Optimization on Rosenbrock Function')
plt.legend()
plt.show()