function save_to_csv(cost_d,cost_e,areas)

    G = graph(cost_d);     
    x_cc = arrayfun(@(ver) ver.pos.x, areas); 
    x_cc = reshape(x_cc, length(x_cc),1);
    y_cc = arrayfun(@(ver) ver.pos.y, areas); 
    y_cc = reshape(y_cc, length(y_cc),1);
    cities = arrayfun(@(city) city.contains_cues, areas);
    ids = arrayfun(@(ver) ver.area_id, areas); 
    ids = reshape(ids,length(ids), 1); 
    C = ids(cities == 1); 

    Y = [x_cc, y_cc, ids]; 
    edges = G.Edges.EndNodes;
    %Table 1. 
    T = array2table(cost_d) ;
    numColumns = length(areas); 
    T.Properties.VariableNames = string(1:numColumns);
    writetable(T,'distance_cost.csv');
    
    %Table 2. 
    T = array2table(cost_e); 
    T.Properties.VariableNames = string(1:numColumns); 
    writetable(T,'energy_cost.csv'); 
    
    %Table 3. 
    T = array2table(Y); 
    T.Properties.VariableNames(1:3) = {'X_coords','Y_coords','Area_id'}; 
    writetable(T,'areas.csv'); 
    
    %Table 4. 
    T = array2table(C); 
    T.Properties.VariableNames(1:1) ={'Customers ids'}; 
    writetable(T,'customers.csv'); 

    %Table 5. 
    T = array2table(edges); 
    T.Properties.VariableNames(1:2) = {'i','j'}; 
    writetable(T,'edges.csv'); 
    
    
end

