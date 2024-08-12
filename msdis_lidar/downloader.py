import requests
import json
import geopandas as gpd
import pandas as pd
from rich.progress import Progress, TextColumn, BarColumn
from rich import print


class LidarDownloader:

    ## Written by M. Jaweed Nazary
    ## University of Missouri-Columbia 
    ## July 2024
    def __init__(self, base_url):
        self.base_url = base_url
        ## Note: The base URL is the URL of the ArcGIS Feature Service that contains the LiDAR data

    def fetch_lidar_download_links(self, polygons):
        # Initialize an empty GeoDataFrame to store the results
    
        gdf_lidar = gpd.GeoDataFrame()

        total_polygons = len(polygons)  # Total number of polygons

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>6.2f}%"),
            TextColumn("[progress.completed]{task.completed}/{task.total}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Fetching LiDAR Download Links", total=total_polygons)

            for index, poly in enumerate(polygons):
                progress.update(task, description=f"Fetching LiDAR Download Links for polygon {index + 1} of {total_polygons}")

                # Convert polygon coordinates to a string format required by the ArcGIS API
                polygon_string = json.dumps({
                    "rings": [poly],
                    "spatialReference": {"wkid": 4326}
                })


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
                        
            
                base_url = "https://services2.arcgis.com/kNS2ppBA4rwAQQZy/ArcGIS/rest/services/MO_LAS_Tiles/FeatureServer/0/query?"
                
                # Construct the query URL
                query_params = {
                    "geometryType": "esriGeometryPolygon",
                    "geometry": polygon_string,
                    "inSR": 4326,
                    "spatialRel": "esriSpatialRelIntersects",
                    "outFields": ",".join(fields),
                    "returnGeometry": "true",
                    "outSR": 4326,
                    "f": "json"
                }
                query_url = base_url + "&".join([f"{k}={v}" for k, v in query_params.items()])

                # Send GET request
                response = requests.get(query_url)

                # Check if request was successful
                if response.status_code == 200:
                    data = response.json()

                    if 'features' in data and data['features']:
                        # Prepare a URL for exporting the data to GeoJSON
                        export_params = query_params.copy()
                        export_params["f"] = "geojson"
                        export_url = base_url + "&".join([f"{k}={v}" for k, v in export_params.items()])

                        # Send GET request to export URL
                        export_response = requests.get(export_url)

                        # Check if export request was successful
                        if export_response.status_code == 200:
                            # Load the data into GeoDataFrame
                            export_data = export_response.json()
                            new_data = gpd.GeoDataFrame.from_features(export_data['features'])

                            # Append the new data to the existing GeoDataFrame
                            if not gdf_lidar.empty:
                                gdf_lidar = pd.concat([gdf_lidar, new_data], ignore_index=True)
                            else:
                                gdf_lidar = new_data
                else:
                    # Handle errors silently
                    pass

                progress.update(task, advance=1)

        print(f"Downloaded links for {len(gdf_lidar)} LiDAR tiles have been fetched successfully.")
        return gdf_lidar
