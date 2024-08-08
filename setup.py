from setuptools import setup, find_packages

setup(
    name='Automatic_LiDAR_Download_from_MSDIS',
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
    url='https://github.com/JaweedNazary/Automatic_LiDAR_Download_from_MSDIS',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',  # Add your supported Python versions here
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6',  # Specify the minimum Python version required
)

