#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from setuptools import setup, find_packages

setup(
    name='msdis_lidar',
    version='0.1.0',
    description='Download and manage LiDAR data from MSDIS',
    author='M. Jaweed Nazary',
    author_email='jaweedpy@gmail.com',
    packages=find_packages(),
    install_requires=[
        'geopandas',
        'rich',
        'requests',
        'folium',
        'tabulate'
    ],
    entry_points={
        'console_scripts': [
            'msdis-lidar=msdis_lidar.cli:main',
        ],
    },
)

