from setuptools import setup, find_packages
#from TreeDetection import __version__

setup(
    name="TreeDetection",
    version=__version__,
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
