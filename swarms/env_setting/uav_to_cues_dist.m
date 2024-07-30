function distance = uav_to_cues_dist(swarm,num_cues,cues)

    distance = zeros(swarm.size,num_cues);
    cue_x_coords = cues.pos(:,1:num_cues); 
    cue_y_coords = cues.pos(:,num_cues + 1:end); 
    for ii = 1:swarm.size
        for jj = 1:length(cues.area_id)
            if swarm.uavs(ii).current_state.area_id == cues.area_id(jj)
                for cue =  1:num_cues
                    cue_pos = [cue_x_coords(jj,cue), cue_y_coords(jj,cue)];
                    distance(ii,cue) = calculate_distance("U2C",swarm.uavs(ii).current_state,[],cue_pos); 
                end 
            end 
        end
    end 
end

