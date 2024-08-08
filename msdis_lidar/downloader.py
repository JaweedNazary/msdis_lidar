import json
import requests
import geopandas as gpd
from rich.progress import Progress, TextColumn, BarColumn
from rich.console import Console

class LidarDownloader:
    def __init__(self, polygons):
        self.polygons = polygons
        self.gdf_lidar = gpd.GeoDataFrame()

    def get_download_links(self):
        total_polygons = len(self.polygons)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>6.2f}%"),
            TextColumn("[progress.completed]{task.completed}/{task.total}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Fetching LiDAR Download Links", total=total_polygons)
            for index, poly in enumerate(self.polygons):
                progress.update(task, description=f"Fetching LiDAR Download Links for polygon {index + 1} of {total_polygons}")
                polygon_string = self._format_polygon_string(poly)
                query_url = self._construct_query_url(polygon_string)
                response = requests.get(query_url)
                if response.status_code == 200:
                    new_data = self._process_response(response)
                    if new_data is not None:
                        self.gdf_lidar = self.gdf_lidar.append(new_data, ignore_index=True)
                progress.update(task, advance=1)
        print(f"Downloaded links for {len(self.gdf_lidar)} LiDAR tiles have been fetched successfully.")
        return self.gdf_lidar

    def _format_polygon_string(self, polygon):
        return json.dumps({
            "rings": [polygon],
            "spatialReference": {"wkid": 4326}
        })

    def _construct_query_url(self, polygon_string):
        base_url = "https://services2.arcgis.com/kNS2ppBA4rwAQQZy/ArcGIS/rest/services/MO_LAS_Tiles/FeatureServer/0/query?"
        query_params = {
            "geometryType": "esriGeometryPolygon",
            "geometry": polygon_string,
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": ",".join(self._get_fields()),
            "returnGeometry": "true",
            "outSR": 4326,
            "f": "json"
        }
        query_url = base_url + "&".join([f"{k}={v}" for k, v in query_params.items()])

        return query_url, query_params

    def _get_fields(self):
        return [
            "FID", "OBJECTID", "Name_1", "ftp_1", "Acq_Year_1", 
            "ftp", "Project", "updated_ft", "Shape_Leng", 
            "Shape_Area", "Shape__Area", "Shape__Length"
        ]

    def _process_response(self, response, query_params):
        data = response.json()
        if 'features' in data and data['features']:
            export_params = query_params.copy()
            export_params["f"] = "geojson"
            export_url = base_url + "&".join([f"{k}={v}" for k, v in export_params.items()])
            export_response = requests.get(export_url)
            if export_response.status_code == 200:
                export_data = export_response.json()
                return gpd.GeoDataFrame.from_features(export_data['features'])
        return None

    def download_files(self, urls, names):
        console = Console()
        with Progress() as progress:
            total_files = len(urls)
            for index, (url, name) in enumerate(zip(urls, names)):
                self._download_file(url, name, progress, total_files, index)
                console.log(f"Downloaded: {name}")

    def _download_file(self, url, output_path, progress, total_files, index):
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 8192
            task = progress.add_task(f"Downloading {index + 1}/{total_files}: {os.path.basename(output_path)}", total=total_size)
            with open(output_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    file.write(chunk)
                    progress.update(task, advance=len(chunk))
                progress.update(task, completed=total_size)

