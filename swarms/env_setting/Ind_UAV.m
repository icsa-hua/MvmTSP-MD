classdef Ind_UAV
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        uav_id
        group_id 
        current_state
        next_state
        max_battery
        min_battery
        attribute
        v_hor
        v_ver
    end
    
    methods
        function obj = Ind_UAV(uav_id, group_id, area_id, pos, height, battery, time, v_hor, v_ver)
            %UNTITLED Construct an instance of this class
            %   Detailed explanation goes here
            obj.uav_id = uav_id;
            obj.group_id = group_id; 
            obj.current_state.area_id = area_id; 
            obj.current_state.x = pos.x;
            obj.current_state.y = pos.y;
            obj.current_state.z = height;
            obj.current_state.time = time;
            obj.current_state.battery = battery; 
            obj.current_state.rate = 0; 
            obj.current_state.comm_type = "None";
            obj.current_state.sinr = 0;
            obj.current_state.action = "None"; 
            obj.next_state = obj.current_state;
            obj.max_battery = 1500.0;
            obj.min_battery = 150.0;
            obj.attribute = "follower"; 
            obj.v_hor = v_hor; 
            obj.v_ver = v_ver; 
        end
           
        function hover(obj,completed_missions)
            obj.next_state = obj.current_state; 
            [cost, time] = calculate_energy('Hov',obj,[],completed_missions);
            obj.next_state.battery = obj.next_state - cost; 
            obj.next_state.action = 'Hov'; 
            
        end
    end
end

