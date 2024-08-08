#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# msdis_lidar/tests/test_shapefile_handler.py
import pytest
from msdis_lidar.shapefile_handler import ShapefileHandler

def test_get_polygon_string():
    handler = ShapefileHandler("Perche_Creek_HU12.shp")
    polygons = handler.get_polygon_string()
    
    assert len(polygons) > 0, "Polygons list should not be empty"

