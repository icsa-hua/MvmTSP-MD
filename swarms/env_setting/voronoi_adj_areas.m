function adjRegions = voronoi_adj_areas(locations, point,C,num_neigh)
    
    %Find Region
    
    
    target_locations = zeros(length(locations),2); 
    for ii = 1:length(locations)
        target_locations(ii,1) = locations(ii).x; 
        target_locations(ii,2) = locations(ii).y; 
    end 
    
    target_point(1,1) = point.x; 
    target_point(1,2) = point.y; 
    targetRegion = dsearchn(target_locations,target_point);
    
    %Find Adjacent Regions 
    targetVertices = C{targetRegion};
    adjRegions = zeros(1,num_neigh);
    counter = 1; 
    for i = 1:length(C)
        if counter >= num_neigh + 1
            break; 
        end 
        if i ~= targetRegion && any(ismember(targetVertices,C{i}))
           adjRegions(counter) = i; 
           counter = counter + 1; 
           
        end
    end 
   
    
end

