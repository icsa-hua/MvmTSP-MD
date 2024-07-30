function plotVoronoiRegion(area_centroids, voronoi_region)
    % Plot the Voronoi region associated with the leader index

    hold on;

    % Plot the Voronoi cells
    voronoi(area_centroids(:, 1), area_centroids(:, 2));

    % Plot the leader's Voronoi region
    plot(voronoi_region(:, 1), voronoi_region(:, 2), 'r-', 'LineWidth', 2);

    % Plot the area centroids
    scatter(area_centroids(:, 1), area_centroids(:, 2), 'bo', 'filled');

    % Customize the plot as needed
    title('Voronoi Diagram with Leader''s Region');
    xlabel('X-coordinate');
    ylabel('Y-coordinate');

    hold off;
end