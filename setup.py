from setuptools import setup, find_packages

setup(
    name="dsge",
    version="0.1.0",
    description="Smets and Wouters DSGE Model Implementation with Extensions",
    author="Research Team",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "pandas_datareader",
        "matplotlib",
        "seaborn",
        "pymc",
        "sympy",
        "numba",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "mypy",
            "sphinx",
        ],
        "jax": ["jax", "jaxlib"],
    },
    python_requires=">=3.8",
)