import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# 1. Setup Parameters
N = 50             # Larger lattice since it's faster now!
T = 2.27            # Critical Temperature
J = 1.0
num_sweeps = 20000

@njit
def fast_metropolis(lattice, T, N, J, num_sweeps):
    """JIT-compiled Metropolis algorithm with strict typing for lookup table."""
    # Use floats for the keys (4.0, 8.0) to match the type of delta_E
    prob_lookup = {
        4.0: np.exp(-4.0 * J / T),
        8.0: np.exp(-8.0 * J / T)
    }
    
    mag_history = np.zeros(num_sweeps)
    energy_history = np.zeros(num_sweeps)
    
    # Initial magnetization and energy
    m = np.sum(lattice)
    e = 0.0
    for i in range(N):
        for j in range(N):
            e += -J * lattice[i, j] * (lattice[(i+1)%N, j] + lattice[i, (j+1)%N])

    for s in range(num_sweeps):
        for _ in range(N * N):
            i = np.random.randint(0, N)
            j = np.random.randint(0, N)
            spin = lattice[i, j]
            
            nb = (lattice[(i+1)%N, j] + lattice[(i-1)%N, j] +
                  lattice[i, (j+1)%N] + lattice[i, (j-1)%N])
            
            delta_E = 2.0 * J * spin * nb
            
            accept = False
            if delta_E <= 0:
                accept = True
            else:
                # delta_E is positive; lookup its probability
                if np.random.random() < prob_lookup[delta_E]:
                    accept = True
            
            if accept:
                lattice[i, j] *= -1
                m += 2 * lattice[i, j]
                e += delta_E
        
        mag_history[s] = abs(m) / (N * N)
        energy_history[s] = e / (N * N)
        
    return lattice, mag_history, energy_history


# 2. Initialize and Run
lattice = np.random.choice(np.array([1, -1]), size=(N, N))

print(f"Running {num_sweeps} sweeps on {N}x{N} lattice...")
lattice, h_m, h_e = fast_metropolis(lattice, T, N, J, num_sweeps)
print("Done!")

# 3. Visualization
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
ax1.imshow(lattice, cmap='gray'); ax1.set_title(f"Final State (T={T})")
ax2.plot(h_m, color='blue'); ax2.set_title("Magnetization"); ax2.set_xlabel("Sweeps")
ax3.plot(h_e, color='red'); ax3.set_title("Energy"); ax3.set_xlabel("Sweeps")
plt.show()
