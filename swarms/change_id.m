function best_swarm = change_id(swarms,stored_id)

    
    avail_ids = arrayfun(@(y) y.group_id, swarms); 
    remaining_ids = swarms(avail_ids~=stored_id); 
    
    if isempty(remaining_ids)
        error("No ids array was provided"); 
    end 
    
    %if isempty(remaining_ids)
    %    error("No ids array was provided"); 
    %else
    %    random_index = randi([1,length(remaining_ids)],1);
    %    random_id = remaining_ids(random_index); 
    %end 

    best_swarm = 0; 
    best_levels = -1000; 
    for ii = 1:length(swarms)
        energy_level_avg = 0; 
        for jj = 1:swarms(ii).size
            energy_level_avg = energy_level_avg + swarms(ii).uavs(jj).current_state.battery; 
        end 
        energy_level_avg = energy_level_avg / swarms(ii).size; 
        if energy_level_avg > best_levels
            best_levels = energy_level_avg; 
            best_swarm = ii; 
        end 
    end   
    
    
    disp(best_swarm);
    disp(best_levels)
    
    
    
    
    
    
end

