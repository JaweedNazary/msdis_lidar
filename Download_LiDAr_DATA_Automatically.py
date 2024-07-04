#!/usr/bin/env python
# coding: utf-8

## This is code is written by Mohammad Jaweed Nazary 
## All rights are reserved to University of Missouri - Columbia 
## July 2024 

# ### Lets get the required libraries

# In[1]:


import os
import requests
import json


# ### Downloading and Processing `Lidar Data`

# In[36]:


# Change this to your area of interest Bounding box coordinates
bboxcoords1 = "-91.90,39.64,-92.30,39.85"

# List of fields to include in the query
fields = [
    "FID",
    "OBJECTID",
    "Name_1",
    "ftp_1",
    "Acq_Year_1",
    "ftp",
    "Project",
    "updated_ft",
    "Shape_Leng",
    "Shape_Area",
    "Shape__Area",
    "Shape__Length"
]

# Construct the query URL
WebSitetoSearch = "https://services2.arcgis.com/kNS2ppBA4rwAQQZy/ArcGIS/rest/services/MO_LAS_Tiles/FeatureServer/0/query?"
WebSitetoSearch += "geometryType=esriGeometryEnvelope&geometry="
WebSitetoSearch += bboxcoords1
WebSitetoSearch += "&inSR=4326"
WebSitetoSearch += "&spatialRel=esriSpatialRelIntersects"
WebSitetoSearch += "&outFields=" + ",".join(fields)
WebSitetoSearch += "&returnGeometry=true&outSR=4326&f=json"

# Print the query URL
print(WebSitetoSearch)

# Send GET request
response = requests.get(WebSitetoSearch)

# Check if request was successful
if response.status_code == 200:
    data = response.json()
    # Check if there are any features in the response
    if data['features']:
        # Prepare a URL for exporting the data to shapefile
        export_url = "https://services2.arcgis.com/kNS2ppBA4rwAQQZy/ArcGIS/rest/services/MO_LAS_Tiles/FeatureServer/0/query?"
        export_url += "where=1=1&geometryType=esriGeometryEnvelope&geometry="
        export_url += bboxcoords1
        export_url += "&inSR=4326"
        export_url += "&spatialRel=esriSpatialRelIntersects"
        export_url += "&outFields=" + ",".join(fields)
        export_url += "&returnGeometry=true&outSR=4326&f=geojson"

        # Send GET request to export URL
        export_response = requests.get(export_url)

        # Check if export request was successful
        if export_response.status_code == 200:
                    
            print("\n#################################")
            print("# Export request was successful #")
            print("#################################")

        else:
            print("Failed to export data to shapefile.")
    else:
        print("No features found in the specified bounding box.")
else:
    print("Failed to retrieve data from the server.")


# In[37]:


data = json.loads(export_response.content)


# In[38]:


data


# In[39]:


import geopandas as gpd
gdf = gpd.GeoDataFrame.from_features(data['features'])
gdf


# In[40]:

# we can check if the we got the right lidar tiles, this generate a html map using folium to show the LiDAR tiles to be downloaded. 
import folium


m = folium.Map(location = [38.573936, -92.603760], zoom_start=6, tiles='OpenStreetMap')


for idx, row in gdf.iterrows():
    if row['Acq_Year_1'] == 2008:
        color = 'blue'
    elif row['Acq_Year_1'] == 2006:
        color = 'green'
    else:
        color = 'red'
    folium.GeoJson(row['geometry'].__geo_interface__, style_function=lambda x, color=color: {'color': color}).add_to(m)
    
    

# Display the map
m.save('map_with_zoomable_plot.html')

m


# In[41]:


gdf.iloc[0]


# In[42]:


from tabulate import tabulate

data = dict(gdf["Acq_Year_1"].value_counts())

# Convert data to a list of lists for tabulate
table_data = [[year, count] for year, count in data.items()]

# Print in a nice table format
print(tabulate(table_data, headers=['Acquisition Year', 'Number of Tiles'], tablefmt='pretty'))


# In[11]:


for links in gdf["updated_ft"]:
    print(links)

