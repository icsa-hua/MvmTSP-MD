function [dist,ind] = calculate_distance(type,uav,sites,other)
%%%This function is called for multiple distinct cases. 
    %1) 3D Distance between a UAV and the GCS (U2I). 
    %2) 3D Distance between a UAV and the rest of the group's UAVs (U2U). 
    %3) 3D Distance between a UAV and the centers of possible areas (Area). 
    %4) 3D Distance between a UAV and the CUE when there is one in the area (U2C). 
    %5) Horizontal distance between two points. 
    %6) Horizontal distance between a UAV and a cellular user (distance between centers - Hor_CUE)

    %-------Parameters----------%
    % - uavs = The UAV/UAVs we want to calculate the distance for.  
    % - gcs_pos = the 3D positions of the gcs/same as a site's center.  
    % - group_size = the number of uavs for the swarm. 
    % - type = Variable to determine the distance we want to calculate. 
    % - leader (~) = the leading UAV of the swarm in order to address the closest BS.
    % - other = the second's object coordinates to be compared. 
    % - uav_map = object array holding the id and swarm position of each
    % drone.     
    
    %-------Returns--------------%
    % - dist = this is the desirable distance for the UAV and a corresponding element.  
    % - ind = this is not always returned as something other than zero. However it indicates which gcs the UAV as a swarm prefers.  
    
    ind = 0; 
    dist = 0; 
    min_dist = 10000; 
    switch type        
        case 'U2I' %Find the distance between the uavs and the closest gcs. 
            swarm = uav; 
            dist_uav_bs = zeros(swarm.size,2); 
            for ii = 1:swarm.size
                for jj = 1:length(sites)
                    dist_uav_bs(ii,jj) = sqrt((swarm.uavs(ii).current_state.x - sites(jj).pos.x)^2 + (swarm.uavs(ii).current_state.y - sites(jj).pos.y)^2 + (swarm.uavs(ii).current_state.z- sites(jj).gcs_height)^2);
                    if swarm.uavs(ii).uav_id == swarm.leader.uav_id
                        if min_dist >= dist_uav_bs(ii,jj)
                            min_dist = dist_uav_bs(ii,jj); 
                            ind = jj; 
                        end 
                    end                 
                end 
            end 
            dist = dist_uav_bs;
            clear dist_bs min_dist i j 
            
        case 'U2U' %Find the distance between uavs. 
            swarm = uav; 
            u2u_distances = zeros(swarm.size,swarm.size); 
            for ii = 1:swarm.size
                for jj = 1:swarm.size
                    if ii == jj %Same uav Individual
                        continue; 
                    else
                        u2u_distances(ii,jj) = sqrt((swarm.uavs(jj).current_state.x - swarm.uavs(ii).current_state.x)^2 + (swarm.uavs(jj).current_state.y - swarm.uavs(ii).current_state.y)^2 + (swarm.uavs(jj).current_state.z - swarm.uavs(ii).current_state.z)^2);
                    end 
                end 
            end 
            %u2u_distances = nonzeros(u2u_distances); 
            %u2u_distances = reshape(u2u_distances,[swarm.size-1,swarm.size]); 
            dist = u2u_distances; 
            clear dist_uavs i j
        
        case 'Area' %Find the distance between the position of a uav and its
        %possible areas of affect. 
            dist_uav = zeros(length(other),1);                        
            for i = 1: length(other)                
               dist_uav(i,1) = sqrt((uavs.x_coord - other(i,1))^2 + (uavs.y_coord - other(i,2))^2 + (uavs.height - 0)^2);  
               if min_dist > dist_uav(i)
                   min_dist = dist_uav(i); 
                   ind = i;                  
               end 
            end
            dist = dist_uav;               
            clear dist_uav min_dist i  
        case 'U2C' %3D Distance between UAV-CUE.
            cue_height = 1.5; 
            dist = sqrt((uav.x - other(1,1))^2 + (uav.y-other(1,2))^2 + (uav.z-cue_height)^2); 
        case 'Hor' %Horizontal distance between two elements. 
            dist = sqrt((uav.next_state.x- uav.current_state.x)^2 + (uav.next_state.y-uav.current_state.y)^2);
        case 'Hor_UAV'
            dist = sqrt((uav.next_state.x- uav.current_state.x)^2 + (uav.next_state.y-uav.current_state.y)^2);

        case 'Hor_CUE' %Horizontal distance between UAV-CUE. 
            dist = sqrt((uavs.x_coord - other(1,1))^2 + (uavs.y_coord-other(1,2))^2);

    end   
end

