function relay_comms(swarm)

    swarm.uavs(3).current_state.sinr = 9; 
    SINR_THRESHOLD = 10; 
    relay_uav = zeros(1,swarm.size); 
    for ii = 1:swarm.size 
        current_SINR = swarm.uavs(ii).current_state.sinr; 
        
        if current_SINR < SINR_THRESHOLD
           fprintf('UAV %d has poor communication with SINR: %f dB\n', swarm.uavs(ii).uav_id, current_SINR);
           ids = [swarm.uavs(~ismember([swarm.uavs.uav_id], swarm.uavs(ii).uav_id)).uav_id]; 
           x = arrayfun(@(uav) uav.current_state.x, swarm.uavs(ids));
           x = reshape(x, length(x), 1); 
           y = arrayfun(@(uav) uav.current_state.y, swarm.uavs(ids)); 
           y = reshape(y, length(y), 1); 
           pos = [x y];
           relay_pos = [swarm.uavs(ii).current_state.x swarm.uavs(ii).current_state.y];
           nn = knnsearch(pos,relay_pos); 
           disp(nn);
           relay_uav(ii) = ids(nn);             
        end 
    end 

    
    disp(relay_uav);
    





end

