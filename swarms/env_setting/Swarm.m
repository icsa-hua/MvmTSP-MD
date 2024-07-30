classdef Swarm < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        uavs 
        group_id
        areas_to_visit %path to follow. 
        action
        leader
        energy_levels
        size
        
    end
    
    methods
        function obj = Swarm(uavs,group_id,areas_to_visit, action, leader,energy_levels,size)
            %UNTITLED Construct an instance of this class
            %   Detailed explanation goes here
            obj.uavs = uavs;
            obj.group_id = group_id;
            obj.areas_to_visit = areas_to_visit;
            obj.action = action;
            obj.leader = leader; 
            obj.energy_levels = energy_levels;
            obj.size = size;
            
        end
        
        %-- Stay Functionality 
        function stay(obj,completed_missions)
            disp("Stay Phase....");
            leader_index = 0; 
            for ii = 1:obj.size
                if obj.uavs(ii).uav_id == obj.leader.uav_id
                    leader_index = ii;
                end
               obj.uavs(ii).next_state = obj.uavs(ii).current_state;
               obj.uavs(ii).next_state.time = obj.uavs(ii).current_state.time+1;
               cost = calculate_energy('Stay',obj.uavs(ii),[],completed_missions);
               obj.uavs(ii).next_state.battery = obj.uavs(ii).current_state.battery - cost;
               obj.uavs(ii).next_state.action = 'Stay'; 
           end
           obj.leader = obj.uavs(leader_index); 
           obj.action = 'Stay'; 
        end
        
        %-- Recover (recharge battery) Functionality
        function rec(obj, emerg_land, completed_missions, fin)
            disp("Recovery Phase..."); 
            leader_index = 0; 
            for ii = 1:obj.size 
                %uav = obj.uavs(ii);
                if obj.uavs(ii).uav_id == obj.leader.uav_id
                    leader_index = ii;
                end
                obj.uavs(ii).next_state = obj.uavs(ii).current_state;
                obj.uavs(ii).next_state.battery = obj.uavs(ii).current_state.battery; 
                obj.uavs(ii).next_state.action = 'Rec';
                tmpenergy = calculate_energy('Rec', obj.uavs(ii),[], completed_missions); 
                if tmpenergy >= obj.uavs(ii).max_battery
                    tmpenergy = obj.uavs(ii).max_battery - obj.uavs(ii).next_state.battery; 
                end 
                disp(tmpenergy); 
                if emerg_land == 1
                    obj.uavs(ii).next_state.time = fin; 
                    for jj = obj.uavs(ii).current_state.time:fin
                        obj.uavs(ii).next_state.battery = tmpenergy + obj.uavs(ii).next_state.battery; 
                        if obj.uavs(ii).next_state.battery > obj.uavs(ii).max_battery
                            obj.uavs(ii).next_state.battery = obj.uavs(ii).max_battery; 
                        end 
                    end
                else
                   obj.uavs(ii).next_state.battery = obj.uavs(ii).next_state.battery + tmpenergy; 
                   obj.uavs(ii).next_state.time = obj.uavs(ii).current_state.time + 1; 
                   
                end
            end 
            obj.leader = obj.uavs(leader_index);
            obj.action = 'Rec';
        end 
        
        %-- Ascend (rise from ground into the air) Functionality
        function asc(obj,next_positions,sites,completed_missions)
           disp("Ascend Phase ... "); 
           flag = 0;
           leader_index = 0; 
           for ii = 1: obj.size
              if obj.uavs(ii).uav_id == obj.leader.uav_id
                 pos_ = 1;
                 flag = 1; 
                 leader_index = ii; 
              else 
                 x = length(next_positions.x);  
                 if flag == 0
                    pos_ = randi([2,x]);  
                 elseif flag == 1
                    pos_ = randi([1,x]);  
                 end 
              end
                            
              obj.uavs(ii).next_state = obj.uavs(ii).current_state; 
              obj.uavs(ii).next_state.area_id = next_positions.id(pos_); 
              obj.uavs(ii).next_state.x = next_positions.x(pos_); 
              obj.uavs(ii).next_state.y = next_positions.y(pos_); 
              obj.uavs(ii).next_state.z = next_positions.z(pos_);
              obj.uavs(ii).next_state.action = 'Asc'; 
              
              [cost,est_time] = calculate_energy('Asc',obj.uavs(ii),sites,completed_missions);
              est_time_mins = est_time/60; 
              hover_time = 10 - est_time_mins;
              obj.uavs(ii).next_state.battery = obj.uavs(ii).next_state.battery  - cost;
              obj.uavs(ii).next_state.time = obj.uavs(ii).current_state.time + 1;

              next_positions.id(pos_) = []; 
              next_positions.x(pos_) = []; 
              next_positions.y(pos_) = [];
              next_positions.z(pos_) = [];
                              
           end 
           obj.leader = obj.uavs(leader_index); 
           obj.action = 'Asc'; 
        end
        
        
        %-- Surveillance Functionality
        function surv(obj,depots,u2c_dists, completed_missions)
            
            leader_index = 1; 
            for ii = 1:obj.size
                
                if obj.uavs(ii).uav_id == obj.leader.uav_id
                     leader_index = ii;
                end
                
                obj.uavs(ii).next_state = obj.uavs(ii).current_state;
                [cost,est_time] = calculate_energy('Surv',obj.uavs(ii),[],completed_missions);
                obj.uavs(ii).next_state.battery = obj.uavs(ii).next_state.battery - cost;
                obj.uavs(ii).next_state.action = 'Surv'; 
            end 
            
            [rate,sinr] = calculate_rate('U2I',obj,depots,[],[],[],u2c_dists);
            
            for uav = 1:obj.size
                obj.uavs(uav).next_state.sinr = sinr(uav); 
                obj.uavs(uav).next_state.rate = rate(uav); 
                obj.uavs(uav).next_state.comm_type = 'U2I'; 
            end 
            obj.leader = obj.uavs(leader_index);           
            obj.action = 'Surv'; 
        end 
        
        function move(obj,completed_missions)
            for uav = 1:obj.size
                obj.uavs(uav).next_state = obj.uavs(uav).current_state; 
                if next_position.x == obj.uavs(uav).current_state.x && next_position.y == obj.uavs(uav).current_state.y
                    obj.uavs(uav).hover(completed_missions);
                end 
                obj.uavs(uav).next_state.x = next_position.x;
                obj.uavs(uav).next_state.y = next_position.y;
                obj.uavs(uav).next_state.z = next_position.z;
                
                
                
            end 
        end 
        
        
        %-- Coverage Functionality 
        function cov(obj, u2c_dists, completed_missions)
            leader_index = 1; 
            for uav = 1:obj.size
                if obj.uavs(uav).uav_id == obj.leader.uav_id
                     leader_index = uav;
                end
                obj.uavs(uav).next_state = obj.uavs(uav).current_state; 
                [cost, ~] = calculate_energy('Cov', obj.uavs(uav),[], completed_missions); 
                obj.uavs(uav).next_state.battery = obj.uavs(uav).next_state.battery - cost; 
                obj.uavs(uav).next_state.action = 'Surv'; 
            end 
            
            [cue_rate, sinr] = calculate_rate('U2C', obj,[],[],[],[],u2c_dists); 
            for uav = 1:obj.size
                obj.uavs(uav).next_state.sinr = sinr(uav); 
                obj.uavs(uav).next_state.rate = cue_rate(uav); 
                obj.uavs(uav).next_state.comm_type = 'U2C'; 
            end 
            obj.leader = obj.uavs(leader_index);
            obj.action = 'Cov';
            
        end 
        
        %-- Descend (Return to charge station) Functionality
        function desc(obj,depots,completed_missions)
            disp("Descend Phase");
            leader_index = 1; 
            for ii = 1:obj.size
                if obj.uavs(ii).uav_id == obj.leader.uav_id
                    leader_index = ii;
                end
               obj.uavs(ii).next_state.x = depots.x; 
               obj.uavs(ii).next_state.y = depots.y; 
               obj.uavs(ii).next_state.z = depots.gcs_height; 
               cost = calculate_energy('Desc',obj.uavs(ii),depots,completed_missions);
               obj.uavs(ii).next_state.battery = obj.uavs(ii).current_state.battery - cost; 
               obj.uavs(ii).next_state.action = 'Desc'; 
               obj.uavs(ii).next_state.sinr = obj.uavs(ii).current_state.sinr; 
               obj.uavs(ii).next_state.rate = obj.uavs(ii).current_state.rate; 
               obj.uavs(ii).next_state.comm_type = obj.uavs(ii).current_state.comm_type; 
            end
            obj.leader = obj.uavs(leader_index);
        end
        
        %-- Update States (synchronize the timestep with the current state)
        %Executed at every time step. 
        function update_state(obj)
            leader_index = 1; 
            nrgy_levels = zeros(1,obj.size); 
            for ii = 1:obj.size
                if obj.uavs(ii).uav_id == obj.leader.uav_id
                    leader_index = ii;
                end
                nrgy_levels(ii) = obj.uavs(ii).next_state.battery;
                obj.uavs(ii).current_state = obj.uavs(ii).next_state;
            end 
            obj.leader = obj.uavs(leader_index);
            obj.energy_levels = nrgy_levels;
        end 
        
        %-- UAV-to_UAV (enable u2u comms, before every action on the air) Functionality
        function [u2c_distances, u2u_distances] = u2u_enabled(obj, comm_type, depots,areas,cues, num_cues)
            u2c_distances = uav_to_cues_dist(obj,num_cues,cues);
            u2u_distances = calculate_distance(comm_type,obj,depots,[]); 
            [rate,sinr] = calculate_rate(comm_type,obj,depots, areas, cues, u2u_distances,u2c_distances);
            
            for ii = 1:obj.size
                obj.uavs(ii).current_state.rate = rate(ii);
                obj.uavs(ii).current_state.sinr = sinr(ii);
                obj.uavs(ii).current_state.comm_type = 'U2U'; 
            end
            
            %relay_comms(obj); %Use another UAV as a relay for the not
            %sufficient communication of a member.
            
            
            
            
            
            
            
        end
        
        
        
    end
end

