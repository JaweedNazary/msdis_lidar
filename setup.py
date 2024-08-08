from setuptools import setup, find_packages

setup(
    name='msdis_lidar',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'geopandas',
        'rich',
        'requests',
        'folium',
        'tabulate',
    ],
    author='M. Jaweed Nazary',
    description='A Python package for downloading and visualizing LiDAR data from MSDIS',
    url='https://github.com/yourusername/msdis_lidar',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
