from setuptools import setup, find_packages
from TreeDetection import __version__, __init__
setup(
    name="TreeDetection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'affine',
        'aiofiles',
        'cupy-cuda113',
        'fiona==1.9.6',
        'geopandas',
        'matplotlib',
        'numpy==1.23.0',
        'numba>=0.60.0',
        'numpy',
        'opencv_python',
        'pandas',
        'pycocotools',
        'PyYAML==6.0.2',
        'rasterio',
        'Rtree',
        'scikit_learn',
        'scipy',
        'Shapely==2.0.6',
        'tqdm',
        'detectron2 @ git+https://github.com/facebookresearch/detectron2.git'
    ],
    url='https://github.com/Jonetz/TreeDetection',
    extras_require={
        "dev": [
            "pytest",  # For testing purposes
            "flake8",  # For linting
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.9",
)
