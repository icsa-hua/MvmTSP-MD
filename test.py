import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point
import geopandas as gpd
from geopy.distance import geodesic

# Generate realistic latitude and longitude points (e.g., around Berlin)
num_areas = 100
np.random.seed(0)
latitudes = np.random.uniform(52.3, 52.6, num_areas)
longitudes = np.random.uniform(13.2, 13.6, num_areas)
points = np.column_stack((longitudes, latitudes))  # x=lon, y=lat
print(points)
# Convert to Cartesian for Voronoi
vor = Voronoi(points)

# Create bounding box to clip infinite Voronoi regions
bbox = Polygon([(13.15, 52.25), (13.65, 52.25), (13.65, 52.65), (13.15, 52.65)])

# Create Voronoi polygons and clip to bbox
regions = []
centroids = []
for region_idx in vor.point_region:
    region = vor.regions[region_idx]
    if not -1 in region and len(region) > 0:
        poly_points = [vor.vertices[i] for i in region]
        poly = Polygon(poly_points)
        poly = poly.intersection(bbox)
        if poly.area > 0:
            regions.append(poly)
            centroids.append(poly.centroid.coords[0])

# Compute distance matrix between centroids
n = len(centroids)
distance_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        distance_matrix[i, j] = geodesic((centroids[i][1], centroids[i][0]), (centroids[j][1], centroids[j][0])).km
print(distance_matrix)
# Generate users in each region
users_per_area = 5
user_points = []
for region in regions:
    minx, miny, maxx, maxy = region.bounds
    count = 0
    while count < users_per_area:
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        if region.contains(Point(x, y)):
            user_points.append((x, y))
            count += 1

# voronoi_plot_2d(vor)

# Plotting
from mpl_toolkits.mplot3d import Axes3D

# Assign synthetic altitudes to centroids (e.g., 100m ± 20m)
altitudes = np.random.uniform(80, 120, len(centroids))

# 3D Plotting
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot Voronoi region edges in 3D at z=0
for poly in regions:
    x, y = poly.exterior.xy
    z = np.zeros_like(x)
    ax.plot(x, y, z, color='gray', alpha=0.5)

# Plot centroids with altitude
for i, (lon, lat) in enumerate(centroids):
    ax.scatter(lon, lat, altitudes[i], c='red', marker='x', s=50)
    ax.text(lon, lat, altitudes[i] + 5, f'A{i}', color='black')

# Plot user points at ground level (z=0)
for (x, y) in user_points:
    ax.scatter(x, y, 0, c='green', s=10)

# Plot drone paths between centroids
for i in range(len(centroids)):
    for j in range(i+1, len(centroids)):
        x_vals = [centroids[i][0], centroids[j][0]]
        y_vals = [centroids[i][1], centroids[j][1]]
        z_vals = [altitudes[i], altitudes[j]]
        ax.plot(x_vals, y_vals, z_vals, 'blue', alpha=0.3)

# Axis labels
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Altitude (m)')
ax.set_title('3D Voronoi Areas with Drone Paths and Users')
plt.tight_layout()
plt.show()
# fig, ax = plt.subplots(figsize=(10, 10))
# for poly in regions:
#     x, y = poly.exterior.xy
#     ax.fill(x, y, alpha=0.4, edgecolor='black')

# ax.scatter(*zip(*centroids), c='red', marker='x', label='Centroids')
# ax.scatter(*zip(*points), c='blue', marker='o', label='Original Nodes')
# ax.scatter(*zip(*user_points), c='green', s=10, label='Users')

# ax.set_xlabel("Longitude")
# ax.set_ylabel("Latitude")
# ax.set_title("Voronoi Areas with Users")
# ax.legend()
# plt.grid(True)
# plt.show()

# Optional: print distance matrix
print("\nDistance Matrix (km):")
print(np.round(distance_matrix, 2))
