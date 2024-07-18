function past_uavs = update_table(past_uavs,swarm,fin)
%UPDATE_TABLE Summary of this function goes here
%   Detailed explanation goes here
    
    leader = swarm.leader;
    action = leader.next_state.action;
    if strcmp(action,'Rec')
        for jj = 1:swarm.size
            uav = swarm.uavs(jj); 
            cost = uav.next_state.battery - uav.current_state.battery;
            cost = round((cost / fin)); 
            id = uav.uav_id; 
            for ii = uav.current_state.time:fin
                past_uavs(id,ii).uav = uav; 
                past_uavs(id,ii).cost = cost; 
                past_uavs(id,ii).action = action;
                past_uavs(id,ii).rate = uav.next_state.rate; 
                past_uavs(id,ii).sinr = uav.next_state.sinr; 
                past_uavs(id,ii).comm_type = uav.next_state.comm_type;
            end 
        end 
        
    else
        for ii = 1:swarm.size 
            uav = swarm.uavs(ii);                 
            id = uav.uav_id; 
            moment = uav.next_state.time; 
            past_uavs(id,moment).uav = uav; 
            past_uavs(id,moment).cost = abs(uav.next_state.battery - uav.current_state.battery); 
            past_uavs(id,moment).action = action; 
            past_uavs(id,moment).rate = uav.next_state.rate; 
            past_uavs(id,moment).sinr = uav.next_state.sinr; 
            past_uavs(id,moment).comm_type = uav.next_state.comm_type;
        end 
    end   
        
end

