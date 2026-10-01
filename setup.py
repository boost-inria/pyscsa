# setup.py
"""
Setup configuration for SCSA library.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyscsa",
    version="1.0.0",
    author="BOOST",
    author_email="boost@inria.fr",
    description="Semi-Classical Signal Analysis for signal and image processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/boost-inria/pyscsa",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Signal Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "scikit-learn>=0.23.0",
        "matplotlib>=3.2.0",
        "pandas>=1.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "numpydoc>=1.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "scsa-demo=pyscsa.cli:main",
        ],
    },
)
