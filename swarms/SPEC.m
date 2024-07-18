classdef SPEC
    %SPECS This class is used to hold all the important information when
    %executing the program. 
    %It stores the uav as an object of UAV, with info such as position and
    %battery, it stores the action for every time slot, the rate when
    %communications are involved and the communication type at the same
    %extend. 
    
    properties
        uav
        action       
        rate
        comm_type
        cost
        sinr
    end
    
    methods
        function obj = SPEC(uav,action,rate,comm_type,cost,sinr)
            if nargin > 0 
                obj.uav = uav; 
                obj.action = action;                 
                obj.rate = rate;
                obj.comm_type = comm_type;
                obj.cost = cost;
                obj.sinr = sinr;
                
            end 
        end 
        function obj = register(obj,uav,action,rate,comm_type,cost,sinr)

            obj.uav = uav; 
            obj.action = action;                 
            obj.rate = rate;
            obj.comm_type = comm_type;
            obj.cost = cost;
            obj.sinr = sinr;
        end 
        
    end
end

