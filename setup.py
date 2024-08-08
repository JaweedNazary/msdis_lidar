from setuptools import setup, find_packages

setup(
    name='msdis_lidar',
    version='0.1',
    packages=find_packages(include=['msdis_lidar', 'msdis_lidar.*']),
    install_requires=[
        'geopandas',
        'rich',
        'requests',
        'folium',
        'tabulate',
    ],
    tests_require=[
        'pytest',  # Add pytest as a test dependency
    ],
    test_suite='tests',
    include_package_data=True,
    zip_safe=False,
    author='M. Jaweed Nazary',
    description='A Python package for downloading and visualizing LiDAR data from MSDIS',
    url='https://github.com/JaweedNazary/Automatic_LiDAR_Download_from_MSDIS',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.6',  # Specify the minimum Python version required
)

