# Set this file as the working directory
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Import necesary libraries
import numpy as np; from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "cm"})
plt.rcParams.update({"lines.linewidth": 0.8, "lines.markersize": 0.8})
plt.rcParams['axes.labelsize'] = 12 

# ----------------------------------------------------------
# Main Functions: Forest Game Implementation
# ----------------------------------------------------------
def rainforest(L, S0 = 0.1, p = 0.5):
    """Defines a random initial state of the forest. That is a square lattice of size LxL with a probability p of having a tree of size S0 in each cell.

    Args:
        L (int): Lattice size
        S0 (float): Minimum tree size
        p (float): Probability of having a tree of size S0 in each cell

    Returns:
        np.array((L,L)): Lattice with the initial state of the forest
    """    
    lattice = np.zeros((L, L))

    # Create a mask where each cell has a probability (p) of being True
    growth_mask = np.random.random((L, L)) <= p

    # Set cells in the lattice to S0 where the growth mask is True
    lattice[growth_mask] = S0

    return lattice

def birth(lattice, p_birth, S0):
    """Defines the birth rule for the forest. That is, a tree of size S0 is born with probability p_birth in each empty cell.

    Args:
        lattice (np.matrix): forest lattice
        p_birth (float): probability of birth
        S0 (float): minimum tree size

    Returns:
        np.matrix: forest lattice after applying the birth rule
    """    
    # Create a mask for empty cells
    empty_mask = (lattice == 0)

    # Generate a random array of the same shape as the lattice
    random_values = np.random.random(lattice.shape)

    # Create a birth mask where cells are empty and the random value is below the birth threshold
    birth_mask = empty_mask & (random_values <= p_birth)

    # Set cells in the lattice to S0 where the birth mask is True
    lattice[birth_mask] = S0

    return lattice

def heaviside_vectorized(mu, gamma, lattice):
    """heaviside_vectorized calculates the Heaviside function values for each cell in the lattice. This accounts for \Detla n_{ij} in the growth rule.

    Args:
        mu (float): parameter
        gamma (float): interaction strength
        lattice (np.matrix): forest lattice

    Returns:
        np.matrix: Heaviside function values for each cell in the lattice
    """
    # Define the kernel for neighbor sum
    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])
    neighbors_sum = convolve2d(lattice, kernel, mode='same', boundary='wrap')

    # Calculate the Heaviside function values
    heaviside_values = mu - (gamma / 8) * neighbors_sum  
    heaviside_values[heaviside_values < 0] = 0 # Set negative values to zero

    return heaviside_values

def growth(lattice, S0, Sc, mu, gamma):
    """Defines the growth rule for the forest. 

    Args:
        lattice (np.matrix): forest lattice
        S0 (float): minimum tree size
        Sc (float): maximum tree size
        mu (float): parameter
        gamma (float): interaction strength

    Returns:
        np.matrix: forest lattice after applying the growth rule
    """
        
    # Create a mask for cells that are eligible for growth
    growth_mask = (lattice <= Sc) & (lattice >= S0)

    # Calculate the Heaviside values for the entire lattice
    heaviside_values = heaviside_vectorized(mu, gamma, lattice)

    # Apply growth only to eligible cells
    lattice[growth_mask] += heaviside_values[growth_mask]

    return lattice

def find_nth_moore_layer(lattice, i, j, n):
    """Find the nth layer of Moore neighbors in a square lattice, i.e, a square (2n+1)x(2n+1) excluding the inner (2n-1)x(2n-1) square. (see https://mathworld.wolfram.com/MooreNeighborhood.html for more details).

    Args:
        i (int): x coordinate of the cell.
        j (int): y coordinate of the cell.
        n (int): n-th Moore neighborhood.
        grid_size (int): The size of the grid.

    Returns:
        list: list of tuples with the coordinates of the nth Moore neighbors.
    """
    L = lattice.shape[0]
    neighbors = []
    for dx in range(-n, n+1):
        for dy in range(-n, n+1):
            if max(abs(dx), abs(dy)) == n:  # This condition excludes inner squares
                # Wrap around the grid using modulo operator to maintain periodic boundaries
                neighbors.append(((i + dx) % L, (j + dy) % L))

    return neighbors

def find_neighbors_at_R(lattice, i, j, R):
    """Finds the neighbors of a cell exactly at a radius R. Here we consider euclidian distance.

    Args:
        lattice (np.matrix): forest lattice
        i (int): x coordinate of the cell
        j (int): y coordinate of the cell
        R (float): radius

    Returns:
        list: list of tuples with the coordinates of the neighbors
    """
    n_rows, n_cols = lattice.shape
    max_offset = int(np.ceil(R))  # Maximum offset to consider

    # Create arrays for row and column offsets
    row_offsets = np.arange(-max_offset, max_offset + 1)
    col_offsets = np.arange(-max_offset, max_offset + 1)

    # Calculate the grid of distances considering periodic boundaries
    row_distances = np.minimum(np.abs(row_offsets), n_rows - np.abs(row_offsets))**2
    col_distances = np.minimum(np.abs(col_offsets), n_cols - np.abs(col_offsets))**2
    grid_distances = np.sqrt(row_distances[:, np.newaxis] + col_distances)

    # Find neighbors exactly at the radius R
    neighbor_mask = grid_distances == R
    neighbor_offsets = np.argwhere(neighbor_mask) - max_offset

    # Calculate neighbor coordinates with periodic boundary conditions
    neighbors = np.mod(np.array([i, j]) + neighbor_offsets, [n_rows, n_cols])

    return list(map(tuple, neighbors))

def generate_radius_list(max_radius):
    """Generates a list of radii to consider from 1 to max_radius.

    Args:
        max_radius (float): maximum radius

    Returns:
        np.array: array of radii
    """    
    radius_set = set()
    for x in range(max_radius + 1):
        for y in range(x + 1):
            r2 = x**2 + y**2 
            if r2 <= max_radius**2:
                radius_set.add(r2)
    return np.sqrt(sorted(radius_set))[1:] # Remove the first element (R = 0)

def gap_form(lattice, i, j, radius_list, Moore = False):
    """Applies the gap formation rule to a cell. That is, if the sum of the trees in the neighborhood is less than or equal to the tree in the cell, then the trees in the neighborhood die. The radius of the neighborhood is defined by the inequality.

    Args:
        lattice (np.matrix): forest lattice
        i (int): x coordinate of the cell
        j (int): y coordinate of the cell
        radius_list (np.array): array of radii to consider
        Moore (bool, optional): Flag to choose which distance to use in gap formation. Defaults to False.

    Returns:
        np.matrix: forest lattice after applying the gap formation rule
    """    
    S = lattice[i, j]
    S_nn = 0
    for R in radius_list:
        neighbors = find_nth_moore_layer(lattice, i, j, R) if Moore else find_neighbors_at_R(lattice, i, j, R)
        xs, ys = list(zip(*neighbors)) # From [(x1,y1), ...] to [x1, ...], [y1, ...]
        S_nn += np.sum(lattice[xs, ys])
        if S_nn <= S:
            lattice[xs, ys] = 0
        else:
            break

    return lattice

def death(lattice, pd, Sc, Moore = False):
    """Defines the death rule for the forest. A given tree dies randomly with probability pd. A tree that reaches the maximum size Sc also dies.

    Args:
        lattice (np.matrix): forest lattice
        pd (float): probability of death
        Sc (float): maximum tree size
        Moore (bool, optional): Flag to choose which distance to use in gap formation. Defaults to False.

    Returns:
        np.matrix: forest lattice after applying the death rule
    """    
    
    L = lattice.shape[0]
    
    # Create masks for cells above Sc and cells that die randomly
    above_sc_mask = (lattice >= Sc)
    death_mask = np.random.random(lattice.shape) <= pd
    gap_mask = above_sc_mask | (lattice > 0) & death_mask # combined mask
    gap_indices = np.argwhere(gap_mask) # array of indices of cells that will form gaps

    # Generate a list of radii to consider
    max_radius = L // 2  # Maximum radius is half the size of the grid
    radius_list = [i for i in range(1, max_radius + 1)] if Moore else generate_radius_list(max_radius)
    
    # Apply gap_form to the necessary cells
    for i, j in gap_indices: 
        lattice = gap_form(lattice, i, j, radius_list, Moore)
    
    # Set to zero the cells that died
    lattice[above_sc_mask] = 0
    lattice[(lattice > 0) & death_mask] = 0
    
    return lattice

# ----------------------------------------------------------
# Calculations
# ----------------------------------------------------------
Moore = False # Flag to use Moore neighborhood

# Simulation in time
def simulation_in_time(lattice, time_steps, S0, Sc, pb, pd, mu, gamma, Moore = False):
    """Simulates the forest dynamics in time.

    Args:
        lattice (np.matrix): forest lattice
        time_steps (int): number of time steps
        S0 (float): Minimum tree size
        Sc (float): Maximum tree size
        pb (float): Probability of birth
        pd (float): Probability of death
        mu (float): Parameter
        gamma (float): Interaction strength
        Moore (bool, optional): Flag to choose which distance to use in gap formation. Defaults to False.

    Returns:
        np.array: forest lattice after applying the rules time_steps times
    """    
    for _ in range(time_steps):
        lattice = growth(lattice, S0, Sc, mu, gamma)
        lattice = death(lattice, pd, Sc, Moore)
        lattice = birth(lattice, pb, S0)
    return lattice
    
# Sanity Checks: Biomass & Fourier Transform
# ----------------------------------------------------------

# 1. Biomass (Fig.5a)
def biomass(lattice, pd, steps = 700, S0 = 0.1, Sc = 30, mu = 1, gamma = 1, pb = 0.3, Moore = False):
    """Calculates the biomass of the forest.

    Args:
        lattice (np.matrix): forest lattice
        steps (int): number of steps

    Returns:
        np.array: array of biomass values in time
    """    
    
    biomass = np.zeros(steps); time_steps = np.arange(steps)
    for i in time_steps:
        biomass[i] = np.sum(lattice) # biomass is the sum of all trees in the lattice
        lattice = growth(lattice, S0, Sc, mu, gamma)
        lattice = death(lattice, pd, Sc, Moore)
        lattice = birth(lattice, pb, S0)
    return time_steps, biomass

def plot_biomass(t_steps, biom, save=False): 
    """
    Plot biomass over time.

    Parameters:
    - t_steps (array-like): Array of time steps.
    - biom (array-like): Array of biomass values corresponding to each time step.
    - save (bool, optional): Whether to save the plot as a PDF file. Default is False.

    Returns:
    None

    The function creates a plot of biomass against time. If save is set to True, the plot will be saved as 'biomass.pdf' in the current directory.
    """
    fig, ax = plt.subplots()
    ax.plot(t_steps, biom, color="k")
    ax.set_xlabel("Time") 
    ax.set_ylabel("Biomass") 
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    if save: plt.savefig('biomass.pdf', dpi=300)
    
    plt.show()
    
t_steps, biom = biomass(rainforest(40), 0.01, Moore = Moore)
# plot_biomass(t_steps[100:701], biom[100:701])
# plot_biomass(t_steps, biom) # In order to see the comment of p.38 about the steep increase and final stabilization of the biomass at n = 50

# 2. Fourier Transform (Fig.5b)
def plot_fourier_transform(biomass, steps = 700, save=False):
    """
    Plot the Fourier transform of biomass data.

    Parameters:
    - biomass (array-like): Biomass data.
    - steps (int, optional): Number of steps for the Fourier transform. Default is 700.
    - save (bool, optional): Whether to save the plot as a PDF file. Default is False.

    Returns:
    None

    This function calculates the Fourier transform of the biomass data and plots the amplitude against frequency in a logarithmic scale. 
    If save is set to True, the plot will be saved as 'fourier_transform.pdf' in the current directory.
    """
    transform = np.fft.rfft(biomass) # one-dimensional discrete Fourier Transform for real input (https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html)
    # Return the Discrete Fourier Transform sample frequencies for real input
    frequencies = np.fft.rfftfreq(steps, d=1/steps) # d=1/steps sets the sample spacing

    fig, ax = plt.subplots()
    ax.loglog(frequencies, np.abs(transform), color="black") 
    ax.set_xlabel("Frequency"); ax.set_ylabel("Amplitude") 

    if save: plt.savefig('fourier_transform.pdf', dpi=300)
    
    plt.show()
    
# plot_fourier_transform(biom)

# Fractal measures (Fig.7)
# ----------------------------------------------------------
def Dq(lattice, l, qs, epsilon):
    """Dq calculates the correlation dimension of order q for a given lattice as defined in equation (3) of the paper.

    Args:
        lattice (np.matrix): forest lattice
        l (int): size of the box
        qs (np.array): array of q values
        epsilon (float): box size

    Raises:
        ValueError: l must be divisible by L for the box embeding to fit the lattice

    Returns:
        np.array: array of Dq values
    """    
    Np = np.count_nonzero(lattice == 0) # Total number of trees with size 0

    def Nj(lattice, L, l):
        """Nj calculates the distribution function N(j) for a given lattice as defined in the Appendix of the paper.

        Returns:
            dict: dictionary with key-value pairs of the number of boxes with j zeros
        """        
        if L % l != 0: raise ValueError('L must be divisible by l')

        # Dictionary of the number of boxes with j zeros (distribution function N(j))
        Njs = {j: 0 for j in range(1, l**2 + 1)}

        for i in range(0, L, l):
            for j in range(0, L, l):
                # Select the box of size lxl
                box = lattice[i:i+l, j:j+l]
                
                # Count the number of zeros in the box
                number_of_zeros = np.count_nonzero(box == 0)

                if number_of_zeros > 0:
                    Njs[number_of_zeros] += 1

        return Njs
    
    def Xq(qs, Njs, Np):
        """Xq calculates X(q) for a given lattice as defined in the Appendix of the paper.

        Args:
            Np (int): Total number of trees with size 0

        Returns:
            np.array: array of Xq values
        """        
        Xqs = np.zeros(len(qs))
        for i, q in enumerate(qs):
            Xqs[i] = np.sum([Njs[j] * (j / Np)**q for j in Njs])
        return Xqs
    
    L  = len(lattice); Njs = Nj(lattice, L, l); Xqs = Xq(qs, Njs, Np)
    
    return (1 / (qs - 1)) * np.log10(Xqs)/np.log10(epsilon)

def falpha(Dqs, qs):
    """falpha calculates the alpha and the spectrum of fractal dimensions (f) via equations (6) and (7) of the paper respectively. Notice that the formula used here is - f(alpha) of the paper which has a typo as one might appreciate in the later paper Self-organized Criticality in Rainforest Dynamics.

    Args:
        Dqs (np.array): array of Dq values
        qs (np.array): array of q values

    Returns:
        tuple of np.arrays: alpha and f_alpha
    """    
    aux = (qs - 1) * Dqs
    alpha = np.gradient(aux, qs) # numerical derivative of aux with respect to qs
    
    f_alpha = qs * alpha - aux

    return alpha, f_alpha


def plot_fractal(L, l, pd_values, qs, epsilon = 1/20, S0 = 0.1, Sc = 30, p = 0.5, pb = 0.3, mu = 1, gamma = 1, save=False, Moore = False):
    """
    Plot fractal analysis results.

    Parameters:
    - L (int): Size of the lattice.
    - l (int): Length scale for box-counting.
    - pd_values (list): List of values for probability of deforestation.
    - qs (array-like): Array of q values for multifractal analysis.
    - epsilon (float, optional): Epsilon value for multifractal analysis. Default is 1/20.
    - S0 (float, optional): Initial tree size. Default is 0.1.
    - Sc (int, optional): Carrying capacity. Default is 30.
    - p (float, optional): Probability of tree reproduction. Default is 0.5.
    - pb (float, optional): Probability of tree birth. Default is 0.3.
    - mu (float, optional): Rate of tree growth. Default is 1.
    - gamma (float, optional): Rate of deforestation. Default is 1.
    - save (bool, optional): Whether to save the plot as a PDF file. Default is False.
    - Moore (bool, optional): Whether to use Moore neighborhood for simulation. Default is False.

    Returns: None
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    
    for pd in pd_values:
        lattice = rainforest(L, S0, p)
        lattice = simulation_in_time(lattice, 100, S0, Sc, pb, pd, mu, gamma, Moore) # thermalization
        Dqs = Dq(lattice, l, qs, epsilon)
        alpha, f_alpha = falpha(Dqs, qs)

        axs[0].scatter(alpha, f_alpha, s=10)
        axs[0].set_xlabel(r'$\alpha$')
        axs[0].set_ylabel(r'$f(\alpha)$')
        axs[0].xaxis.set_minor_locator(AutoMinorLocator()); axs[0].yaxis.set_minor_locator(AutoMinorLocator())
        
        axs[1].scatter(qs, Dqs, s=10, label=rf'$p_d = {pd}$')
        axs[1].set_xlabel(r'$q$')
        axs[1].set_ylabel(r'$D(q)$')
        axs[1].xaxis.set_minor_locator(AutoMinorLocator()); axs[1].yaxis.set_minor_locator(AutoMinorLocator())

    axs[1].legend()
    plt.tight_layout()

    if save: plt.savefig('fractal_plot.pdf', dpi=300, bbox_inches='tight')

    plt.show()

# plot_fractal(80, 4, [0.005, 0.01, 0.03], np.arange(-7.5, 7.5, 0.1), Moore = Moore)