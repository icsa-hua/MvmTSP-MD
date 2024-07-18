classdef Configuration < handle
    %UNTITLED3 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        number_of_agents
        num_cues
        num_groups
        group_size
        TS
        gcs_height
        height
        areas
        cues_pos
        V
        C
        depots
        actions
        colors
        swarms
        cur_swarms

    end
    
    methods
        function obj = Configuration(number_of_agents, num_cues,group_size, gcs_height, height)
            
            obj.number_of_agents = number_of_agents;
            if isempty(obj.number_of_agents)
                obj.number_of_agents = 3; 
            end 
            obj.num_cues = num_cues; 
            obj.gcs_height = gcs_height; 
            obj.height = height; 
            
            obj.num_groups = obj.number_of_agents ^2; 
            obj.group_size = group_size;
            
            obj.TS = linspace(0,144,145); 
            
            [obj.areas,obj.cues_pos,obj.V,obj.C] = region_generation(obj.num_cues,obj.gcs_height); 
            
            obj.depots = arrayfun(@(x) strcmp(x.class_type,'site'),obj.areas); 
            obj.depots = obj.areas(obj.depots); 
            
            color1 = {0.27 0.55 0.863};
            color2 = {0.0 0.4470 0.7410};
            color3 = {0.9290 0.6940 0.1250};
            color4 = {0.128 0.128 0.128};
            obj.colors = {color1,color2,color3,color4}; 
            
            actions_set = ["Stay";"Rec";"Asc";"Surv";"Move";"Hov";"Cov";"Desc"];
            actions_num = [1;2;3;4;5;6;7;8]; 
            obj.actions = containers.Map(actions_num,actions_set);
        end
        
        function print_environment(obj,option)
           plot_environment_LL(obj.areas,obj.cues_pos.pos(:,1:obj.num_cues),obj.cues_pos.pos(:,obj.num_cues+1:end),option,obj.gcs_height); 
        end 
        
        function [distanceMatrix, timeMatrix, energyMatrix] = objective_function(obj)
           
            x_cc = arrayfun(@(area) area.pos.x, obj.areas); 
            x_cc = reshape(x_cc, length(x_cc),1);
            y_cc = arrayfun(@(area) area.pos.y, obj.areas); 
            y_cc = reshape(y_cc, length(y_cc),1);
            cc = [x_cc, y_cc]; 
            
            distances = pdist(cc,'euclidean');
            distanceMatrix = squareform(distances); %meters. 
            
            g = 9.81; %Acceleration of gravity in m/s^2. 
            p = 0.2; %Density of the air in m^2.  
            a = 1.225; %Rotor disk area in Kg/m^3. 
            lambda = 0.08; %Coefficient for the drag profile depending on the type of the UAV.     
            m = 6;
            num_rot = 4; 
            
            %Time step for calculation. 
            dt = 10; %mins 
            timeMatrix = dt * 60; %seconds
            velocityMatrix = distanceMatrix/timeMatrix; 
            
            E_hov = sqrt((m*g)/(2*p*a));
            W = (m*g)^2;
            K = sqrt(2)*p*a;
            M = 1; 
            D = sqrt(velocityMatrix.^2 + sqrt(velocityMatrix.^4 + 4*(E_hov^4)));
            E_h = (W/K).*(M./D).*timeMatrix;
            E_r = (lambda*p*a.*(velocityMatrix.^3).*timeMatrix)./8;
            
            E_move = E_h + E_r; 
            E_move = E_move./timeMatrix; %Power of unit, or Watts
            T = g * m;
            
            P = (T^(3/2)/(sqrt(2*num_rot*p*a))); %This is in W = J/s
            energy = E_move + P;
            energyMatrix = energy .* (1/6);
            
        end 
        
        
        
        
        function obj = classify_swarms(obj,v_hor,v_ver)
            depot = obj.depots(1); 
            pos.x = 0; 
            pos.y = 0; 
            uavs = repmat(Ind_UAV(0,0,0,pos,0,0,0,0,0),1,(obj.num_groups*obj.group_size)); 
            flag = 1; 
            group_id = 1; 
            for ii = 1:obj.num_groups
                leader = randi([flag,flag+obj.group_size-1], 1); 
                for jj = 1:obj.group_size
                    uavs(flag) = Ind_UAV(flag, group_id,depot.area_id, depot.pos,0,1500,1,v_hor,v_ver); 
                    if leader == flag
                        uavs(flag).attribute = 'leader'; 
                    end 
                    flag = flag + 1; 
                end 
                group_id = group_id + 1; 
            end 
            keyword = 'leader'; 
            obj.cur_swarms = generateRandomNumbers(obj.number_of_agents, obj.num_groups); 
            obj.swarms = repmat(Swarm(uavs(1), 0, [], "None",0,[],obj.group_size), 1, obj.num_groups); 
            leaders = find(strcmp(arrayfun(@(y)  y.attribute, uavs,'UniformOutput', false), keyword ));
            unvis_areas = arrayfun(@(x) x.area_id, obj.areas); 
            for ii = 1:obj.num_groups
                startind = ii*obj.group_size-obj.group_size+1; 
                endind =  ii*obj.group_size; 
                obj.swarms(ii) = Swarm(uavs(startind:endind),uavs(startind).group_id, unvis_areas,"None",uavs(leaders(ii)),[],obj.group_size);
            end 
            
        end
        
        
        
        
        
    end
end

