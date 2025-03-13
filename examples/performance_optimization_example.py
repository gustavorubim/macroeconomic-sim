#!/usr/bin/env python
"""
Performance Optimization Example

This script demonstrates techniques for improving computational performance in DSGE models:
1. Numba JIT compilation for critical functions
2. Parallel processing for simulations
3. JAX integration for GPU acceleration
4. Memory optimization strategies
5. Performance benchmarking

The example shows how to optimize performance for large-scale or time-sensitive applications.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from functools import wraps
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import tracemalloc

# Check if numba is available
try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Numba not available. JIT compilation examples will be skipped.")

# Check if JAX is available
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("JAX not available. GPU acceleration examples will be skipped.")


# Utility function for timing
def timeit(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds to run")
        return result
    return wrapper


# 1. Standard implementation (baseline)
class StandardSolver:
    """Standard implementation of a DSGE model solver."""
    
    def __init__(self, model_size=10):
        """
        Initialize the solver.
        
        Args:
            model_size (int): Size of the model (number of state variables)
        """
        self.model_size = model_size
        
        # Initialize model matrices
        # In a real implementation, these would be derived from the model
        # For demonstration, we'll use random matrices
        np.random.seed(42)
        self.A = np.random.randn(model_size, model_size)
        self.B = np.random.randn(model_size, model_size)
        self.C = np.random.randn(model_size, model_size)
        self.D = np.random.randn(model_size, model_size)
        
        # Ensure stability
        self.A = 0.95 * self.A / np.max(np.abs(np.linalg.eigvals(self.A)))
        
        print(f"Initialized standard solver with model size {model_size}")
    
    @timeit
    def solve(self):
        """
        Solve the model using standard methods.
        
        Returns:
            tuple: (P, F) policy and transition matrices
        """
        # In a real implementation, this would solve the model
        # For demonstration, we'll use a simplified algorithm
        
        # Solve for policy function P
        P = np.linalg.solve(self.C, -self.D @ self.B)
        
        # Solve for transition function F
        F = self.A + self.B @ P
        
        return P, F
    
    @timeit
    def simulate(self, periods=100, num_simulations=1):
        """
        Simulate the model.
        
        Args:
            periods (int): Number of periods to simulate
            num_simulations (int): Number of simulations to run
            
        Returns:
            list: List of simulated paths
        """
        # Solve the model if not already solved
        P, F = self.solve()
        
        # Run simulations
        simulations = []
        for i in range(num_simulations):
            # Initialize state
            state = np.zeros(self.model_size)
            
            # Simulate
            states = np.zeros((periods, self.model_size))
            controls = np.zeros((periods, self.model_size))
            
            for t in range(periods):
                # Add random shock
                if t > 0:
                    state = F @ state + 0.01 * np.random.randn(self.model_size)
                
                # Compute control variables
                control = P @ state
                
                # Store results
                states[t] = state
                controls[t] = control
            
            simulations.append((states, controls))
        
        return simulations


# 2. Numba-optimized implementation
if HAS_NUMBA:
    class NumbaOptimizedSolver:
        """Numba-optimized implementation of a DSGE model solver."""
        
        def __init__(self, model_size=10):
            """
            Initialize the solver.
            
            Args:
                model_size (int): Size of the model (number of state variables)
            """
            self.model_size = model_size
            
            # Initialize model matrices
            # In a real implementation, these would be derived from the model
            # For demonstration, we'll use random matrices
            np.random.seed(42)
            self.A = np.random.randn(model_size, model_size)
            self.B = np.random.randn(model_size, model_size)
            self.C = np.random.randn(model_size, model_size)
            self.D = np.random.randn(model_size, model_size)
            
            # Ensure stability
            self.A = 0.95 * self.A / np.max(np.abs(np.linalg.eigvals(self.A)))
            
            print(f"Initialized Numba-optimized solver with model size {model_size}")
        
        @timeit
        def solve(self):
            """
            Solve the model using Numba-optimized methods.
            
            Returns:
                tuple: (P, F) policy and transition matrices
            """
            # Use the JIT-compiled function
            return self._solve_jit(self.A, self.B, self.C, self.D)
        
        @staticmethod
        @numba.jit(nopython=True)
        def _solve_jit(A, B, C, D):
            """
            JIT-compiled function to solve the model.
            
            Args:
                A, B, C, D: Model matrices
                
            Returns:
                tuple: (P, F) policy and transition matrices
            """
            # Solve for policy function P
            P = np.linalg.solve(C, -D @ B)
            
            # Solve for transition function F
            F = A + B @ P
            
            return P, F
        
        @timeit
        def simulate(self, periods=100, num_simulations=1):
            """
            Simulate the model using Numba-optimized methods.
            
            Args:
                periods (int): Number of periods to simulate
                num_simulations (int): Number of simulations to run
                
            Returns:
                list: List of simulated paths
            """
            # Solve the model if not already solved
            P, F = self.solve()
            
            # Run simulations
            simulations = []
            for i in range(num_simulations):
                # Use the JIT-compiled function
                states, controls = self._simulate_jit(P, F, periods, self.model_size)
                simulations.append((states, controls))
            
            return simulations
        
        @staticmethod
        @numba.jit(nopython=True)
        def _simulate_jit(P, F, periods, model_size):
            """
            JIT-compiled function to simulate the model.
            
            Args:
                P: Policy matrix
                F: Transition matrix
                periods: Number of periods
                model_size: Size of the model
                
            Returns:
                tuple: (states, controls) simulated paths
            """
            # Initialize state
            state = np.zeros(model_size)
            
            # Simulate
            states = np.zeros((periods, model_size))
            controls = np.zeros((periods, model_size))
            
            for t in range(periods):
                # Add random shock
                if t > 0:
                    state = F @ state + 0.01 * np.random.randn(model_size)
                
                # Compute control variables
                control = P @ state
                
                # Store results
                states[t] = state
                controls[t] = control
            
            return states, controls


# 3. Parallel implementation
class ParallelSolver:
    """Parallel implementation of a DSGE model solver."""
    
    def __init__(self, model_size=10):
        """
        Initialize the solver.
        
        Args:
            model_size (int): Size of the model (number of state variables)
        """
        self.model_size = model_size
        
        # Initialize model matrices
        # In a real implementation, these would be derived from the model
        # For demonstration, we'll use random matrices
        np.random.seed(42)
        self.A = np.random.randn(model_size, model_size)
        self.B = np.random.randn(model_size, model_size)
        self.C = np.random.randn(model_size, model_size)
        self.D = np.random.randn(model_size, model_size)
        
        # Ensure stability
        self.A = 0.95 * self.A / np.max(np.abs(np.linalg.eigvals(self.A)))
        
        print(f"Initialized parallel solver with model size {model_size}")
    
    @timeit
    def solve(self):
        """
        Solve the model.
        
        Returns:
            tuple: (P, F) policy and transition matrices
        """
        # The solution method itself is not parallelized
        # In a real implementation, this might use parallel methods for large models
        
        # Solve for policy function P
        P = np.linalg.solve(self.C, -self.D @ self.B)
        
        # Solve for transition function F
        F = self.A + self.B @ P
        
        return P, F
    
    @staticmethod
    def _simulate_worker(args):
        """
        Worker function for parallel simulation.
        
        Args:
            args: Tuple of (P, F, periods, model_size, seed)
            
        Returns:
            tuple: (states, controls) simulated paths
        """
        P, F, periods, model_size, seed = args
        
        # Set random seed
        np.random.seed(seed)
        
        # Initialize state
        state = np.zeros(model_size)
        
        # Simulate
        states = np.zeros((periods, model_size))
        controls = np.zeros((periods, model_size))
        
        for t in range(periods):
            # Add random shock
            if t > 0:
                state = F @ state + 0.01 * np.random.randn(model_size)
            
            # Compute control variables
            control = P @ state
            
            # Store results
            states[t] = state
            controls[t] = control
        
        return states, controls
    
    @timeit
    def simulate(self, periods=100, num_simulations=1):
        """
        Simulate the model using parallel processing.
        
        Args:
            periods (int): Number of periods to simulate
            num_simulations (int): Number of simulations to run
            
        Returns:
            list: List of simulated paths
        """
        # Solve the model if not already solved
        P, F = self.solve()
        
        # Prepare arguments for parallel execution
        args_list = [(P, F, periods, self.model_size, i) for i in range(num_simulations)]
        
        # Run simulations in parallel
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            simulations = list(executor.map(self._simulate_worker, args_list))
        
        return simulations


# 4. JAX-accelerated implementation
if HAS_JAX:
    class JaxAcceleratedSolver:
        """JAX-accelerated implementation of a DSGE model solver."""
        
        def __init__(self, model_size=10):
            """
            Initialize the solver.
            
            Args:
                model_size (int): Size of the model (number of state variables)
            """
            self.model_size = model_size
            
            # Initialize model matrices
            # In a real implementation, these would be derived from the model
            # For demonstration, we'll use random matrices
            key = jax.random.PRNGKey(42)
            self.A = jax.random.normal(key, (model_size, model_size))
            key, subkey = jax.random.split(key)
            self.B = jax.random.normal(subkey, (model_size, model_size))
            key, subkey = jax.random.split(key)
            self.C = jax.random.normal(subkey, (model_size, model_size))
            key, subkey = jax.random.split(key)
            self.D = jax.random.normal(subkey, (model_size, model_size))
            
            # Ensure stability
            eigvals = jnp.linalg.eigvals(self.A)
            self.A = 0.95 * self.A / jnp.max(jnp.abs(eigvals))
            
            # Convert to device arrays
            self.A = jnp.array(self.A)
            self.B = jnp.array(self.B)
            self.C = jnp.array(self.C)
            self.D = jnp.array(self.D)
            
            print(f"Initialized JAX-accelerated solver with model size {model_size}")
        
        @timeit
        def solve(self):
            """
            Solve the model using JAX-accelerated methods.
            
            Returns:
                tuple: (P, F) policy and transition matrices
            """
            # Use JAX's JIT compilation
            return self._solve_jit(self.A, self.B, self.C, self.D)
        
        @staticmethod
        @jax.jit
        def _solve_jit(A, B, C, D):
            """
            JIT-compiled function to solve the model using JAX.
            
            Args:
                A, B, C, D: Model matrices
                
            Returns:
                tuple: (P, F) policy and transition matrices
            """
            # Solve for policy function P
            P = jnp.linalg.solve(C, -D @ B)
            
            # Solve for transition function F
            F = A + B @ P
            
            return P, F
        
        @timeit
        def simulate(self, periods=100, num_simulations=1):
            """
            Simulate the model using JAX-accelerated methods.
            
            Args:
                periods (int): Number of periods to simulate
                num_simulations (int): Number of simulations to run
                
            Returns:
                list: List of simulated paths
            """
            # Solve the model if not already solved
            P, F = self.solve()
            
            # Run simulations
            simulations = []
            for i in range(num_simulations):
                # Generate a random key for this simulation
                key = jax.random.PRNGKey(i)
                
                # Use JAX's vectorized operations
                states, controls = self._simulate_jit(P, F, periods, self.model_size, key)
                
                # Convert to numpy arrays for consistency with other implementations
                simulations.append((np.array(states), np.array(controls)))
            
            return simulations
        
        @staticmethod
        @jax.jit
        def _simulate_jit(P, F, periods, model_size, key):
            """
            JIT-compiled function to simulate the model using JAX.
            
            Args:
                P: Policy matrix
                F: Transition matrix
                periods: Number of periods
                model_size: Size of the model
                key: JAX random key
                
            Returns:
                tuple: (states, controls) simulated paths
            """
            # Generate all random shocks at once
            keys = jax.random.split(key, periods)
            shocks = 0.01 * jax.random.normal(keys[0], (periods, model_size))
            
            # Initialize state
            state = jnp.zeros(model_size)
            
            # Initialize arrays to store results
            states = jnp.zeros((periods, model_size))
            controls = jnp.zeros((periods, model_size))
            
            # Define a single step of the simulation
            def simulation_step(carry, t):
                state, states, controls = carry
                
                # Add random shock (except for the first period)
                shock = jnp.where(t > 0, shocks[t], 0.0)
                state = F @ state + shock
                
                # Compute control variables
                control = P @ state
                
                # Update states and controls
                states = states.at[t].set(state)
                controls = controls.at[t].set(control)
                
                return (state, states, controls), None
            
            # Run the simulation
            (_, states, controls), _ = jax.lax.scan(
                simulation_step,
                (state, states, controls),
                jnp.arange(periods)
            )
            
            return states, controls


# 5. Memory-optimized implementation
class MemoryOptimizedSolver:
    """Memory-optimized implementation of a DSGE model solver."""
    
    def __init__(self, model_size=10):
        """
        Initialize the solver.
        
        Args:
            model_size (int): Size of the model (number of state variables)
        """
        self.model_size = model_size
        
        # Initialize model matrices
        # In a real implementation, these would be derived from the model
        # For demonstration, we'll use random matrices
        np.random.seed(42)
        self.A = np.random.randn(model_size, model_size)
        self.B = np.random.randn(model_size, model_size)
        self.C = np.random.randn(model_size, model_size)
        self.D = np.random.randn(model_size, model_size)
        
        # Ensure stability
        self.A = 0.95 * self.A / np.max(np.abs(np.linalg.eigvals(self.A)))
        
        print(f"Initialized memory-optimized solver with model size {model_size}")
    
    @timeit
    def solve(self):
        """
        Solve the model using memory-optimized methods.
        
        Returns:
            tuple: (P, F) policy and transition matrices
        """
        # Use single precision to reduce memory usage
        A = self.A.astype(np.float32)
        B = self.B.astype(np.float32)
        C = self.C.astype(np.float32)
        D = self.D.astype(np.float32)
        
        # Solve for policy function P
        P = np.linalg.solve(C, -D @ B)
        
        # Solve for transition function F
        F = A + B @ P
        
        return P, F
    
    @timeit
    def simulate(self, periods=100, num_simulations=1):
        """
        Simulate the model using memory-optimized methods.
        
        Args:
            periods (int): Number of periods to simulate
            num_simulations (int): Number of simulations to run
            
        Returns:
            list: List of simulated paths
        """
        # Solve the model if not already solved
        P, F = self.solve()
        
        # Convert to single precision
        P = P.astype(np.float32)
        F = F.astype(np.float32)
        
        # Run simulations
        simulations = []
        for i in range(num_simulations):
            # Initialize state
            state = np.zeros(self.model_size, dtype=np.float32)
            
            # Use a generator to avoid storing all states in memory at once
            def simulate_generator():
                nonlocal state
                for t in range(periods):
                    # Add random shock
                    if t > 0:
                        state = F @ state + 0.01 * np.random.randn(self.model_size).astype(np.float32)
                    
                    # Compute control variables
                    control = P @ state
                    
                    yield state.copy(), control.copy()
            
            # For demonstration, we'll still collect all states and controls
            # In a real memory-constrained application, you would process them on-the-fly
            states = []
            controls = []
            for state, control in simulate_generator():
                states.append(state)
                controls.append(control)
            
            states = np.array(states, dtype=np.float32)
            controls = np.array(controls, dtype=np.float32)
            
            simulations.append((states, controls))
        
        return simulations


def benchmark_solvers(model_sizes, num_simulations=10, periods=100):
    """
    Benchmark different solver implementations.
    
    Args:
        model_sizes (list): List of model sizes to benchmark
        num_simulations (int): Number of simulations to run
        periods (int): Number of periods to simulate
        
    Returns:
        dict: Benchmark results
    """
    results = {
        "model_sizes": model_sizes,
        "standard": {"solve": [], "simulate": []},
        "numba": {"solve": [], "simulate": []},
        "parallel": {"solve": [], "simulate": []},
        "jax": {"solve": [], "simulate": []},
        "memory": {"solve": [], "simulate": []}
    }
    
    for size in model_sizes:
        print(f"\nBenchmarking model size {size}")
        
        # Standard implementation
        print("\nStandard implementation:")
        solver = StandardSolver(model_size=size)
        
        start_time = time.time()
        P, F = solver.solve()
        solve_time = time.time() - start_time
        results["standard"]["solve"].append(solve_time)
        
        start_time = time.time()
        _ = solver.simulate(periods=periods, num_simulations=num_simulations)
        simulate_time = time.time() - start_time
        results["standard"]["simulate"].append(simulate_time)
        
        # Numba-optimized implementation
        if HAS_NUMBA:
            print("\nNumba-optimized implementation:")
            solver = NumbaOptimizedSolver(model_size=size)
            
            start_time = time.time()
            P, F = solver.solve()
            solve_time = time.time() - start_time
            results["numba"]["solve"].append(solve_time)
            
            start_time = time.time()
            _ = solver.simulate(periods=periods, num_simulations=num_simulations)
            simulate_time = time.time() - start_time
            results["numba"]["simulate"].append(simulate_time)
        
        # Parallel implementation
        print("\nParallel implementation:")
        solver = ParallelSolver(model_size=size)
        
        start_time = time.time()
        P, F = solver.solve()
        solve_time = time.time() - start_time
        results["parallel"]["solve"].append(solve_time)
        
        start_time = time.time()
        _ = solver.simulate(periods=periods, num_simulations=num_simulations)
        simulate_time = time.time() - start_time
        results["parallel"]["simulate"].append(simulate_time)
        
        # JAX-accelerated implementation
        if HAS_JAX:
            print("\nJAX-accelerated implementation:")
            solver = JaxAcceleratedSolver(model_size=size)
            
            start_time = time.time()
            P, F = solver.solve()
            solve_time = time.time() - start_time
            results["jax"]["solve"].append(solve_time)
            
            start_time = time.time()
            _ = solver.simulate(periods=periods, num_simulations=num_simulations)
            simulate_time = time.time() - start_time
            results["jax"]["simulate"].append(simulate_time)
        
        # Memory-optimized implementation
        print("\nMemory-optimized implementation:")
        solver = MemoryOptimizedSolver(model_size=size)
        
        start_time = time.time()
        P, F = solver.solve()
        solve_time = time.time() - start_time
        results["memory"]["solve"].append(solve_time)
        
        start_time = time.time()
        _ = solver.simulate(periods=periods, num_simulations=num_simulations)
        simulate_time = time.time() - start_time
        results["memory"]["simulate"].append(simulate_time)
    
    return results


def plot_benchmark_results(results, output_dir="results/performance"):
    """
    Plot benchmark results.
    
    Args:
        results (dict): Benchmark results
        output_dir (str): Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model_sizes = results["model_sizes"]
    
    # Plot solve times
    plt.figure(figsize=(10, 6))
    
    if HAS_NUMBA:
        plt.plot(model_sizes, results["standard"]["solve"], "o-", label="Standard")
        plt.plot(model_sizes, results["numba"]["solve"], "o-", label="Numba")
        plt.plot(model_sizes, results["parallel"]["solve"], "o-", label="Parallel")
        if HAS_JAX:
            plt.plot(model_sizes, results["jax"]["solve"], "o-", label="JAX")
        plt.plot(model_sizes, results["memory"]["solve"], "o-", label="Memory-optimized")
    
    plt.title("Model Solution Time vs. Model Size")
    plt.xlabel("Model Size")
    plt.ylabel("Time (seconds)")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/solve_times.png")
    
    # Plot simulate times
    plt.figure(figsize=(10, 6))
    
    if HAS_NUMBA:
        plt.plot(model_sizes, results["standard"]["simulate"], "o-", label="Standard")
        plt.plot(model_sizes, results["numba"]["simulate"], "o-", label="Numba")
        plt.plot(model_sizes, results["parallel"]["simulate"], "o-", label="Parallel")
        if HAS_JAX:
            plt.plot(model_sizes, results["jax"]["simulate"], "o-", label="JAX")
        plt.plot(model_sizes, results["memory"]["simulate"], "o-", label="Memory-optimized")
    
    plt.title("Model Simulation Time vs. Model Size")
    plt.xlabel("Model Size")
    plt.ylabel("Time (seconds)")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/simulate_times.png")
    
    # Create a summary table
    summary = pd.DataFrame(index=model_sizes)
    
    if HAS_NUMBA:
        summary["Standard (solve)"] = results["standard"]["solve"]
        summary["Numba (solve)"] = results["numba"]["solve"]
        summary["Parallel (solve)"] = results["parallel"]["solve"]
        if HAS_JAX:
            summary["JAX (solve)"] = results["jax"]["solve"]
        summary["Memory (solve)"] = results["memory"]["solve"]
        
        summary["Standard (simulate)"] = results["standard"]["simulate"]
        summary["Numba (simulate)"] = results["numba"]["simulate"]
        summary["Parallel (simulate)"] = results["parallel"]["simulate"]
        if HAS_JAX:
            summary["JAX (simulate)"] = results["jax"]["simulate"]
        summary["Memory (simulate)"] = results["memory"]["simulate"]
    
    # Calculate speedups
    if HAS_NUMBA:
        for size_idx, size in enumerate(model_sizes):
            baseline_solve = results["standard"]["solve"][size_idx]
            baseline_simulate = results["standard"]["simulate"][size_idx]
            
            summary.loc[size, "Numba Speedup (solve)"] = baseline_solve / results["numba"]["solve"][size_idx]
            summary.loc[size, "Parallel Speedup (solve)"] = baseline_solve / results["parallel"]["solve"][size_idx]
            if HAS_JAX:
                summary.loc[size, "JAX Speedup (solve)"] = baseline_solve / results["jax"]["solve"][size_idx]
            summary.loc[size, "Memory Speedup (solve)"] = baseline_solve / results["memory"]["solve"][size_idx]
            
            summary.loc[size, "Numba Speedup (simulate)"] = baseline_simulate / results["numba"]["simulate"][size_idx]
            summary.loc[size, "Parallel Speedup (simulate)"] = baseline_simulate / results["parallel"]["simulate"][size_idx]
            if HAS_JAX:
                summary.loc[size, "JAX Speedup (simulate)"] = baseline_simulate / results["jax"]["simulate"][size_idx]
            summary.loc[size, "Memory Speedup (simulate)"] = baseline_simulate / results["memory"]["simulate"][size_idx]
    
    # Save summary table
    summary.to_csv(f"{output_dir}/benchmark_summary.csv")
    
    # Plot speedups
    plt.figure(figsize=(12, 8))
    
    if HAS_NUMBA:
        plt.subplot(2, 1, 1)
        plt.plot(model_sizes, summary["Numba Speedup (solve)"], "o-", label="Numba")
        plt.plot(model_sizes, summary["Parallel Speedup (solve)"], "o-", label="Parallel")
        if HAS_JAX:
            plt.plot(model_sizes, summary["JAX Speedup (solve)"], "o-", label="JAX")
        plt.plot(model_sizes, summary["Memory Speedup (solve)"], "o-", label="Memory-optimized")
        
        plt.title("Solver Speedup vs. Model Size")
        plt.xlabel("Model Size")
        plt.ylabel("Speedup (x)")
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(model_sizes, summary["Numba Speedup (simulate)"], "o-", label="Numba")
        plt.plot(model_sizes, summary["Parallel Speedup (simulate)"], "o-", label="Parallel")
        if HAS_JAX:
            plt.plot(model_sizes, summary["JAX Speedup (simulate)"], "o-", label="JAX")
        plt.plot(model_sizes, summary["Memory Speedup (simulate)"], "o-", label="Memory-optimized")
        
        plt.title("Simulation Speedup vs. Model Size")
        plt.xlabel("Model Size")
        plt.ylabel("Speedup (x)")
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/speedups.png")


def memory_profile_solvers(model_size=50, periods=100, num_simulations=10):
    """
    Profile memory usage of different solver implementations.
    
    Args:
        model_size (int): Model size to profile
        periods (int): Number of periods to simulate
        num_simulations (int): Number of simulations to run
        
    Returns:
        dict: Memory usage results
    """
    results = {
        "standard": {"peak": 0, "current": 0},
        "numba": {"peak": 0, "current": 0},
        "parallel": {"peak": 0, "current": 0},
        "jax": {"peak": 0, "current": 0},
        "memory": {"peak": 0, "current": 0}
    }
    
    # Standard implementation
    print("\nProfiling standard implementation:")
    tracemalloc.start()
    
    solver = StandardSolver(model_size=model_size)
    P, F = solver.solve()
    _ = solver.simulate(periods=periods, num_simulations=num_simulations)
    
    current, peak = tracemalloc.get_traced_memory()
    results["standard"]["current"] = current / 1024 / 1024  # MB
    results["standard"]["peak"] = peak / 1024 / 1024  # MB
    print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
    
    tracemalloc.stop()
    
    # Numba-optimized implementation
    if HAS_NUMBA:
        print("\nProfiling Numba-optimized implementation:")
        tracemalloc.start()
        
        solver = NumbaOptimizedSolver(model_size=model_size)
        P, F = solver.solve()
        _ = solver.simulate(periods=periods, num_simulations=num_simulations)
        
        current, peak = tracemalloc.get_traced_memory()
        results["numba"]["current"] = current / 1024 / 1024  # MB
        results["numba"]["peak"] = peak / 1024 / 1024  # MB
        print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
        print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
        
        tracemalloc.stop()
    
    # Parallel implementation
    print("\nProfiling parallel implementation:")
    tracemalloc.start()
    
    solver = ParallelSolver(model_size=model_size)
    P, F = solver.solve()
    _ = solver.simulate(periods=periods, num_simulations=num_simulations)
    
    current, peak = tracemalloc.get_traced_memory()
    results["parallel"]["current"] = current / 1024 / 1024  # MB
    results["parallel"]["peak"] = peak / 1024 / 1024  # MB
    print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
    
    tracemalloc.stop()
    
    # JAX-accelerated implementation
    if HAS_JAX:
        print("\nProfiling JAX-accelerated implementation:")
        tracemalloc.start()
        
        solver = JaxAcceleratedSolver(model_size=model_size)
        P, F = solver.solve()
        _ = solver.simulate(periods=periods, num_simulations=num_simulations)
        
        current, peak = tracemalloc.get_traced_memory()
        results["jax"]["current"] = current / 1024 / 1024  # MB
        results["jax"]["peak"] = peak / 1024 / 1024  # MB
        print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
        print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
        
        tracemalloc.stop()
    
    # Memory-optimized implementation
    print("\nProfiling memory-optimized implementation:")
    tracemalloc.start()
    
    solver = MemoryOptimizedSolver(model_size=model_size)
    P, F = solver.solve()
    _ = solver.simulate(periods=periods, num_simulations=num_simulations)
    
    current, peak = tracemalloc.get_traced_memory()
    results["memory"]["current"] = current / 1024 / 1024  # MB
    results["memory"]["peak"] = peak / 1024 / 1024  # MB
    print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
    
    tracemalloc.stop()
    
    return results


def plot_memory_results(results, output_dir="results/performance"):
    """
    Plot memory profiling results.
    
    Args:
        results (dict): Memory profiling results
        output_dir (str): Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    implementations = []
    peak_memory = []
    current_memory = []
    
    if HAS_NUMBA:
        implementations.append("Standard")
        peak_memory.append(results["standard"]["peak"])
        current_memory.append(results["standard"]["current"])
        
        implementations.append("Numba")
        peak_memory.append(results["numba"]["peak"])
        current_memory.append(results["numba"]["current"])
    
    implementations.append("Parallel")
    peak_memory.append(results["parallel"]["peak"])
    current_memory.append(results["parallel"]["current"])
    
    if HAS_JAX:
        implementations.append("JAX")
        peak_memory.append(results["jax"]["peak"])
        current_memory.append(results["jax"]["current"])
    
    implementations.append("Memory-optimized")
    peak_memory.append(results["memory"]["peak"])
    current_memory.append(results["memory"]["current"])
    
    # Plot memory usage
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(implementations))
    width = 0.35
    
    plt.bar(x - width/2, peak_memory, width, label="Peak Memory")
    plt.bar(x + width/2, current_memory, width, label="Current Memory")
    
    plt.xlabel("Implementation")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Memory Usage by Implementation")
    plt.xticks(x, implementations)
    plt.legend()
    plt.grid(True, axis="y")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/memory_usage.png")
    
    # Create a summary table
    summary = pd.DataFrame({
        "Implementation": implementations,
        "Peak Memory (MB)": peak_memory,
        "Current Memory (MB)": current_memory
    })
    
    # Calculate memory savings
    if HAS_NUMBA:
        baseline_peak = results["standard"]["peak"]
        baseline_current = results["standard"]["current"]
        
        summary["Peak Memory Savings (%)"] = 100 * (1 - summary["Peak Memory (MB)"] / baseline_peak)
        summary["Current Memory Savings (%)"] = 100 * (1 - summary["Current Memory (MB)"] / baseline_current)
    
    # Save summary table
    summary.to_csv(f"{output_dir}/memory_summary.csv", index=False)


def main():
    """Main function demonstrating performance optimization techniques."""
    # Create output directory
    os.makedirs("results/performance", exist_ok=True)
    
    print("=== DSGE Model Performance Optimization Example ===")
    
    # 1. Basic demonstration of different implementations
    print("\n1. Demonstrating Different Implementations")
    
    # Standard implementation
    print("\nStandard implementation:")
    std_solver = StandardSolver(model_size=10)
    std_P, std_F = std_solver.solve()
    std_sim = std_solver.simulate(periods=20, num_simulations=1)
    
    # Numba-optimized implementation
    if HAS_NUMBA:
        print("\nNumba-optimized implementation:")
        numba_solver = NumbaOptimizedSolver(model_size=10)
        numba_P, numba_F = numba_solver.solve()
        numba_sim = numba_solver.simulate(periods=20, num_simulations=1)
    
    # Parallel implementation
    print("\nParallel implementation:")
    parallel_solver = ParallelSolver(model_size=10)
    parallel_P, parallel_F = parallel_solver.solve()
    parallel_sim = parallel_solver.simulate(periods=20, num_simulations=1)
    
    # JAX-accelerated implementation
    if HAS_JAX:
        print("\nJAX-accelerated implementation:")
        jax_solver = JaxAcceleratedSolver(model_size=10)
        jax_P, jax_F = jax_solver.solve()
        jax_sim = jax_solver.simulate(periods=20, num_simulations=1)
    
    # Memory-optimized implementation
    print("\nMemory-optimized implementation:")
    memory_solver = MemoryOptimizedSolver(model_size=10)
    memory_P, memory_F = memory_solver.solve()
    memory_sim = memory_solver.simulate(periods=20, num_simulations=1)
    
    # 2. Benchmark different implementations
    print("\n2. Benchmarking Different Implementations")
    model_sizes = [5, 10, 20, 30, 40, 50]
    benchmark_results = benchmark_solvers(model_sizes, num_simulations=5, periods=50)
    
    # Plot benchmark results
    plot_benchmark_results(benchmark_results)
    print(f"Saved benchmark results to results/performance/")
    
    # 3. Memory profiling
    print("\n3. Memory Profiling")
    memory_results = memory_profile_solvers(model_size=30, periods=50, num_simulations=5)
    
    # Plot memory results
    plot_memory_results(memory_results)
    print(f"Saved memory profiling results to results/performance/")
    
    print("\nPerformance optimization example completed successfully.")


if __name__ == "__main__":
    main()