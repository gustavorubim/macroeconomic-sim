# Macroeconomic Simulation Documentation

## Project Overview

A comprehensive Python implementation of the Smets and Wouters (2007) DSGE model with modern extensions. This project features modular design, multiple solution methods, and advanced estimation capabilities.

## Core Documentation

- [Usage Guide](docs/USAGE_GUIDE.md): Detailed instructions for using the model
- [Documentation Roadmap](docs/DOCUMENTATION_ROADMAP.md): Overview of documentation structure
- [Example Files](docs/EXAMPLE_FILES.md): Documentation for example implementations

## Technical Components

### Model Architecture
- Modular design with clear separation of concerns
- Extensive configuration options
- Multiple solution methods
- Advanced estimation capabilities

### Key Features
1. Base Smets-Wouters Model
   - Complete implementation of SW (2007) equations
   - Household utility maximization
   - Firm production and pricing
   - Wage setting
   - Monetary policy rule

2. Model Extensions
   - Financial frictions module
   - Open economy features
   - Fiscal policy enhancements

3. Solution Methods
   - Perturbation methods (1st, 2nd, 3rd order)
   - Projection methods
   - Policy and transition functions

4. Estimation Framework
   - Bayesian estimation (MCMC)
   - Prior distribution configuration
   - Model evaluation tools
   - Convergence diagnostics

5. Analysis Tools
   - Impulse response functions
   - Historical decompositions
   - Variance decompositions
   - Diagnostic tools

6. Forecasting System
   - Point and density forecasts
   - Scenario analysis
   - Uncertainty quantification

## Repository Structure

```
macroeconomic-sim/
├── config/                  # Configuration files
├── data/                   # Data storage
├── docs/                   # Documentation
├── dsge/                   # Main package
├── examples/               # Example scripts
├── tests/                  # Testing framework
├── results/                # Output storage
└── legacy-docs/            # Archived documentation
```

## Installation & Setup

See [Installation Guide](docs/USAGE_GUIDE.md#installation) for detailed setup instructions.

## Testing Framework

The project includes comprehensive testing:
- Unit tests
- Integration tests
- Functional tests
- Performance tests

## Technology Stack

### Core Dependencies
- numpy, scipy: Numerical computing
- pandas, pandas_datareader: Data management
- matplotlib, seaborn: Visualization
- PyMC: Bayesian estimation
- sympy: Symbolic manipulation

### Optional Enhancements
- numba: JIT compilation
- jax: Automatic differentiation and GPU acceleration

## Contributing

For bug reports, feature requests, and contributions, please follow our contribution guidelines and submit through GitHub.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this implementation in your research, please cite our work and the original Smets and Wouters (2007) paper.