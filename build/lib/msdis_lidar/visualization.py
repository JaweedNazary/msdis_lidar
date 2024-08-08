#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import folium
import geopandas as gpd

class Visualizer:
    @staticmethod
    def visualize_gdf(gdf, other_gdf, initial_location=None, zoom_start=10):
        if gdf.empty and other_gdf.empty:
            raise ValueError("Both GeoDataFrames are empty. No data to visualize.")
        initial_location = initial_location or Visualizer._get_initial_location(gdf, other_gdf)
        m = folium.Map(location=initial_location, zoom_start=zoom_start)
        Visualizer._add_gdf_to_map(m, gdf, 'blue')
        Visualizer._add_gdf_to_map(m, other_gdf, 'red')
        folium.LayerControl().add_to(m)
        return m

    @staticmethod
    def _get_initial_location(gdf, other_gdf):
        combined_geom = gdf.geometry.unary_union.union(other_gdf.geometry.unary_union)
        centroid = combined_geom.centroid
        return [centroid.y, centroid.x]

    @staticmethod
    def _add_gdf_to_map(m, gdf, color):
        geojson_data = gdf.to_json()
        folium.GeoJson(
            geojson_data,
            style_function=lambda feature: {
                'fillColor': color,
                'color': color,
                'weight': 1,
                'fillOpacity': 0.5
            }
        ).add_to(m)

