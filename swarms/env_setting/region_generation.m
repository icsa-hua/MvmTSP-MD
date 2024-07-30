function [areas, CUE, vertexAll, cellAll] = region_generation(num_cues, gcs_height)
    % Initialize all output variables
    vertexAll = []; 
    cellAll = []; 
    areas = [];
    CUE = [];

    % Example data for Y
    Y = [7.4921e5,4.4365e6;  7.4892e5, 4.43639e6;  
         7.4859e5, 4.4367e6;  7.486e5,  4.4372e6; 
         7.4889e5, 4.4373e6;  7.4915e5, 4.43725e6;
         7.4927e5, 4.43682e6; 7.489e5,  4.4369e6;
         7.4909e5, 4.4367e6;  7.4933e5, 4.43627e6;
         7.489e5,  4.43617e6; 7.4855e5, 4.43625e6;
         7.4837e5, 4.4364e6;  7.4835e5, 4.437e6;
         7.4845e5, 4.43755e6; 7.48685e5,4.4378e6;
         7.491e5,  4.43781e6; 7.495e5,  4.4374e6;
         7.4943e5, 4.4368e6;  7.4949e5, 4.4364e6; 
         7.4974e5, 4.4367e6;  7.4962e5,  4.4371e6;  
         7.4958e5, 4.4376e6;  7.4921e5, 4.438e6; 
         7.4895e5, 4.4379e6;  7.485e5,  4.4379e6;
         7.483e5,  4.4378e6;  7.481e5,  4.43723e6;
         7.4812e5, 4.4372e6;  7.4828e5, 4.4368e6;
         7.4829e5, 4.4362e6;  7.484e5,  4.437e6;
         7.4869e5, 4.4359e6;  7.4888e5, 4.436e6;
         7.493e5, 4.43575e6;  7.496e5,  4.4361e6;];

    % Check for duplicates 
    [C, ~, ~] = unique(Y, 'rows'); 
    if length(C) ~= length(Y)
        error("It appears that duplicate areas were found. The original matrix cannot contain duplicates."); 
    end

    num_regions = length(Y);
    pos.x = 0.0;
    pos.y = 0.0;
    areas = repmat(Regions(0, 'none', pos, false, 0), 1, num_regions);

    for area = 1:num_regions
        pos.x = Y(area, 1);
        pos.y = Y(area, 2);
        if area == 8 || area == 9
            areas(area) = Regions(area, 'site', pos, false, gcs_height); 
        else
            areas(area) = Regions(area, 'area', pos, false, 0); 
        end
    end 

    booleBounded = zeros(num_regions, 1); 
    [vertexAll, cellAll] = voronoin(Y); 

    main_region = vertexAll(cellAll{15}, :); 
    plotVoronoiRegion(Y, main_region); 
    x_coords = zeros(num_regions, num_cues);
    y_coords = zeros(num_regions, num_cues); 

    for ii = 1:num_regions
        booleBounded(ii) = ~(any((cellAll{ii}) == 1)); 
        if booleBounded(ii)
            xx0 = Y(ii, 1);
            yy0 = Y(ii, 2);
            cells_xx = vertexAll(cellAll{ii}, 1); 
            cells_yy = vertexAll(cellAll{ii}, 2); 
            numb_Tri = length(cells_xx); 
            indexVertex = 1:(numb_Tri + 1); 
            indexVertex(end) = 1; 
            indexVertex1 = indexVertex(1:numb_Tri); 
            indexVertex2 = indexVertex(2:numb_Tri + 1); 
            area_Tri = abs((cells_xx(indexVertex1) - xx0) .* (cells_yy(indexVertex2) - yy0) - (cells_xx(indexVertex2) - xx0) .* (cells_yy(indexVertex1) - yy0)) / 2;
            area_Poly = sum(area_Tri);
            cdf_Tri = cumsum(area_Tri) / area_Poly;
            index_Tri = find(rand(1, 1) <= cdf_Tri, 1);
            indexVertex1 = indexVertex(index_Tri);
            indexVertex2 = indexVertex(index_Tri + 1);
            Tri_xx = [xx0, cells_xx(indexVertex1), cells_xx(indexVertex2)];
            Tri_yy = [yy0, cells_yy(indexVertex1), cells_yy(indexVertex2)];
            for jj = 1:num_cues
                unirand1 = rand(1, 1); 
                unirand2 = rand(1, 1); 
                x_coords(ii, jj) = (1 - sqrt(unirand1)) * Tri_xx(1) + sqrt(unirand1) * (1 - unirand2) * Tri_xx(2) + sqrt(unirand1) * unirand2 * Tri_xx(3); 
                y_coords(ii, jj) = (1 - sqrt(unirand1)) * Tri_yy(1) + sqrt(unirand1) * (1 - unirand2) * Tri_yy(2) + sqrt(unirand1) * unirand2 * Tri_yy(3);                                          
            end  
        end 
    end 

    indexBounded = find(booleBounded == 1); 
    cue_x = x_coords(indexBounded, :);
    cue_y = y_coords(indexBounded, :); 
    for ii = 1:length(indexBounded) 
        if strcmp(areas(indexBounded(ii)).class_type, 'site')
           cue_x(ii,:) = []; 
           cue_y(ii,:) = []; 
           indexBounded(ii) = 0;
           continue
        end
        areas(indexBounded(ii)).contains_cues = true;
    end 
    
    indexBounded(indexBounded==0)=[];     
    CUE.pos = [cue_x(:,:),cue_y(:,:)];
    CUE.area_id = indexBounded; 
    disp(CUE); 
    
end
