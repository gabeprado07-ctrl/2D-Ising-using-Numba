import numpy as np
import csv
from numba import njit

# 1. Setup Parameters
N = 50             # Larger lattice since it's faster now!
T = 2.4            # Critical Temperature
J = 1.0
num_sweeps = 20000
window = 3000
n_trials = 30

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


# 2. Batch Run 30 Trials
csv_path = rf"C:\Users\gabep\Downloads\2dising_T{T}_30trials.csv"
print(f"Running {n_trials} trials of {num_sweeps} sweeps each on {N}x{N} lattice at T={T}...")

with open(csv_path, mode='w', newline='') as out_file:
    writer = csv.writer(out_file)
    writer.writerow(['trial', 'avg_magnetization', 'avg_energy_per_site'])
    
    for trial in range(1, n_trials + 1):
        lattice = np.random.choice(np.array([1, -1]), size=(N, N))
        lattice, h_m, h_e = fast_metropolis(lattice, T, N, J, num_sweeps)
        
        # Calculate averages over the last window
        if num_sweeps >= window:
            mag_window = h_m[-window:]
            energy_window = h_e[-window:]
        else:
            mag_window = h_m
            energy_window = h_e
        
        avg_mag_last = np.mean(mag_window)
        avg_energy_last = np.mean(energy_window)
        
        writer.writerow([trial, float(avg_mag_last), float(avg_energy_last)])
        print(f"Trial {trial} finished: Avg Mag = {avg_mag_last:.6f}, Avg Energy = {avg_energy_last:.6f}")

print(f"Done! CSV written to {csv_path}")