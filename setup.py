"""
Setup script for VisionForge.

Type-Safe Deep Learning Framework for Computer Vision with Pydantic 
configuration and Optuna hyperparameter optimization.
"""
from pathlib import Path
from setuptools import setup, find_packages


def parse_requirements(filename: str) -> list[str]:
    """
    Parse requirements.txt and extract package names.
    
    Filters out:
    - Comments (lines starting with #)
    - Empty lines
    - Section headers
    
    Args:
        filename: Path to requirements file
        
    Returns:
        List of package requirements
    """
    requirements_path = Path(__file__).parent / filename
    
    if not requirements_path.exists():
        return []
    
    with open(requirements_path, 'r', encoding='utf-8') as f:
        return [
            line.strip() 
            for line in f.readlines() 
            if line.strip() 
            and not line.startswith("#")
            and not line.startswith("=")
        ]


# Read long description from README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()


setup(
    name="visionforge",
    version="0.1.0",
    description="Type-Safe Deep Learning Framework for Computer Vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tommaso Russo",
    url="https://github.com/tomrussobuilds/visionforge",
    license="MIT",
    
    # Package discovery
    packages=find_packages(
        include=['orchard', 'orchard.*', 'tools', 'tools.*'],
        exclude=['tests', 'tests.*', 'docs', 'docs.*']
    ),
    
    # Python version requirement
    python_requires=">=3.10",
    
    # Dependencies
    install_requires=parse_requirements('requirements.txt'),
    
    # Optional dependencies
    extras_require={
        'dev': [
            'black>=23.0.0',
            'flake8>=6.0.0',
            'isort>=5.12.0',
            'mypy>=1.0.0',
        ],
        'docs': [
            'sphinx>=7.0.0',
            'sphinx-rtd-theme>=1.3.0',
        ],
    },
    
    # Package metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Keywords for PyPI
    keywords=[
        "deep-learning",
        "computer-vision",
        "pytorch",
        "pydantic",
        "optuna",
        "hyperparameter-optimization",
        "medical-imaging",
        "type-safe",
    ],
    
    # Include package data
    include_package_data=True,
    
    # Entry points 
    entry_points={
        'console_scripts': [
            'visionforge-train=main:main',
            'visionforge-optimize=optimize:main',
        ],
    } if False else {},  # Set to True to enable CLI commands
    
    # Zip safe
    zip_safe=False,
)