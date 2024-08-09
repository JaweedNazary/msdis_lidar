import geopandas as gpd
from rich.progress import track
import builtins
from rich import print


class ShapefileHandler(builtins.object):
    def __init__(self, shapefile_path):
        self.gdf = gpd.read_file(shapefile_path)
        self.reprojected_gdf = None

    def get_polygon_string(self):
        print(f"{len(self.gdf.geometry)} geometries found")   
        if self.gdf.crs.to_string() != "EPSG:4326":
            print(f'CRS = {self.gdf.crs} is not compatible')
            self.reprojected_gdf = self.gdf.to_crs("EPSG:4326")
            print(f"  Projecting to {self.reprojected_gdf.crs} ...")
            print(f"  Simplifying the polygons...")
        else:
            self.reprojected_gdf = self.gdf

        polygons = []
        for i in track(range(len(self.reprojected_gdf.geometry)), description = f"Getting geometries info..."):
            polygon_coords = []
            for coordinates in self.reprojected_gdf.geometry[i].boundary.coords:
                polygon_coords.append([coordinates[0], coordinates[1]])

            nodes = len(polygon_coords)
            if nodes > 30: 
                polygon_coords = self.simplify_polygon(polygon_coords, nodes, i)
            polygons.append(polygon_coords)

        return polygons

    def simplify_polygon(self, polygon_coords, nodes, i):
        sample_every = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
        while nodes >= 30.0: 
            for sample_no in sample_every:
                polygon_coords = []
                n = 0
                for coordinates in self.reprojected_gdf.geometry[i].boundary.coords:
                    n+=1
                    if n %sample_no == 0:
                        polygon_coords.append([coordinates[0], coordinates[1]])
                nodes = len(polygon_coords) 
                if nodes <= 30:
                    break
            if nodes <= 30:
                break
        return polygon_coords

