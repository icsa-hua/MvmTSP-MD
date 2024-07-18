function experiment()

    
    
    clear 
    clc
    close all 
    tic 
    fprintf('-UAV Swarm Simulation\n');
    
    number_of_agents = 5; 
    num_cues = 10;
    group_size = 5; 
    gcs_height = 25; 
    height = 100; 
    v_hor = 5.5; %m/s
    v_ver = 2.78; 
    
    conf = Configuration(number_of_agents,num_cues,group_size,gcs_height,height); 
    conf.print_environment('Real_area_40');
    conf.classify_swarms(v_hor,v_ver); 
    required_visits = randi([1,3], 1, length(conf.areas)); 
    [cost_d, timestep,cost_e] = conf.objective_function(); 
    
    save_to_csv(cost_d, cost_e, conf.areas); 
    
    try 
        if count(py.sys.path, 'dummy_mvmtsp')==0
            insert(py.sys.path,int32(0), 'dummy_mvmtsp/')
        end 
        depots_ids = arrayfun(@(d) py.int(d.area_id), conf.depots, 'UniformOutput', false); 
        D = py.list(depots_ids);
        m = py.int(number_of_agents);
        
        %py.sys.modules.pop("dummy_mvmtsp.matlab_mvmtsp",py.None); 
        mod = py.importlib.import_module("dummy_mvmtsp.matlab_mvmtsp");
        py.importlib.reload(mod);
        
        results = py.dummy_mvmtsp.matlab_mvmtsp.mvmTSP_problem_formulation(m,D); 
        
        
        
        result = py.matlab_mvmtsp.run(); 
        if isa(result, 'py.str')
            matlabString = string(result);  % Convert Python string to MATLAB string
            disp(matlabString);
        else
            disp('Function did not return a string.');
        end
        mod = py.importlib.import_module('matlab_mvmtsp');
        mod = py.importlib.reload(mod);
        
         
        
        results = mod.mvmTSP_problem_formulation(m,D);
    catch e 
        fprintf('Python error occured: \n');
        fprintf('%s\n', e.message);
    end 
    
    testmvmtsp(conf.areas, cost_d,conf.depots)
    %mvmtsp(number_of_agents, cost_d,timestep,cost_e,required_visits, conf.areas, conf.depots); 
    
    %number_of_agents = 3;
    %num_cues = 10; %0 + (20-0).*rand(1,1); 
    %num_groups = number_of_agents ^ 2; 
    %num_groups = input("\nEnter the number of UAV swarms: "); 
    
    %if isempty(num_groups)
    %    num_groups = 4; 
    %    println("\nNo correct value was inputted, default in use"); 
    %end 
    
    %group_size = 5;
    %group_size = input("\nHow many UAVs in each swarm: "); 
    %if isempty(group_size)
    %    group_size = 3; 
    %    println("\nNo correct value was inputted, default in use"); 
    %end 
    
    t = 1; 
    %gcs_height = 25; %meters (height for gcs)
    %[areas,cues_pos,V,C] = region_generation(num_cues,gcs_height); 
    %depots = arrayfun(@(x) strcmp(x.class_type,'site'),areas); 
    
    %depots = areas(depots);
    %depot = depots(1);
    %pos.x = 0; 
    %pos.y = 0; 
    %uavs = repmat(Ind_UAV(0,0,0,pos,0,0,0),1,(num_groups*group_size)); 
    
    %flag = 1;
    %group_id = 1;
    %for ii = 1:num_groups
    %    leader = randi([flag,flag+group_size-1],1);
    %    for jj=1:group_size
    %        uavs(flag) = Ind_UAV(flag,group_id,depot.area_id,depot.pos, 0, 1500, t);
    %        if leader == flag
    %            uavs(flag).attribute = 'leader';     
    %        end 
    %        flag = flag + 1; 
    %    end
    %    group_id = group_id + 1; 
    %end 
        
    %height = 100; %meters 
    %TS = linspace(0,144,145); 
    %completed_missions = 0; 
    %emerg_desc = 0; 
    %color_counter = 1; 
    
    %%%Colors for each swarm
    %color1 = {0.27 0.55 0.863};
    %color2 = {0.0 0.4470 0.7410};
    %color3 = {0.9290 0.6940 0.1250};
    %color4 = {0.128 0.128 0.128};
    %colors = {color1,color2,color3,color4}; 
    
    %actions_set = ["Stay";"Rec";"Asc";"Surv";"Move";"Hov";"Cov";"Desc"];
    %actions_num = [1;2;3;4;5;6;7;8]; 
    %actions = containers.Map(actions_num,actions_set); 
    
    %option = 'Real_area_40';
    %plot_environment_LL(areas,cues_pos.pos(:,1:num_cues),cues_pos.pos(:,num_cues+1:end),option,gcs_height); 
    %prev_state.h = 0; 
    %prev_state.h2 = 0; 
    %prev_state.h3 = 0; 
    
    %past_uavs = repmat(SPEC, length(uavs), length(TS)); 
    %past_uavs(:,1:length(TS)) = SPEC(uavs(1),'Pending/Waiting',0,'None',0,0); 
    
    vis_areas = zeros(1,length(areas));
    unvis_areas = arrayfun(@(x) x.area_id, areas); 
    surv_areas = zeros(1,length(areas));
    cov_areas = surv_areas; 
    keyword = 'leader';
    flag = false;
    
    %What swarms from the group are currently executing the program. 
    cur_swarms = generateRandomNumbers(number_of_agents,num_groups); 
        
    swarms = repmat(Swarm(uavs(1),0,[],"None",0,[],group_size), 1, num_groups);
    leaders = find(strcmp(arrayfun(@(y) y.attribute, uavs,'UniformOutput', false), keyword));
    
    for ii = 1:num_groups
        startind = ii*group_size-group_size+1;
        endind = ii*group_size; 
        swarms(ii) = Swarm(uavs(startind:endind),uavs(startind).group_id, unvis_areas,"None",uavs(leaders(ii)),[],group_size); 
    end
    
    clear color1 color2 color3 color4 actions_set actions_num pos depot leader group_id startind endind uavs
    
    %An einai na trexoume ton mvmTSP prepei na ton trexoume mia fora edw
    %prin apo arxikopoihsh twn energeiwn. 
    
    
    
    
    
    pause(2); 
    while t<=length(TS)
        %Prepei na elegxw an to current group allazei kathe fora. 
        
        %We now have the swarm, the group id, and the leader of that group.     
        swarms(current_group).update_state();  
        
        clp =  swarms(current_group).leader.current_state.area_id; 
        if strcmp(areas(clp).class_type, 'site')
            if any(swarms(current_group).energy_levels <= swarms(current_group).leader.min_battery)
                %Swarm need to replenish its energy 
                tmp_group = current_group; 
                current_group = change_id(swarms.group_id,tmp_group); 
                fprintf('\nUAV swarms needs to be replaced...');
                fin = t + 6; 
                swarms(tmp_group).rec(emerg_land, completed_missions,fin);
                flag = true; 
                continue; 
            else
                if t~=1 && strcmp(past_uavs(swarm(current_group)).action,'Desc')
                    prob = [0.5,0.5,0,0,0,0,0,0];
                elseif swarms(current_group).leader.current_state.battery == uavs(1).max_battery
                    prob = [0.2,0,0.8,0,0,0,0,0];
                else 
                    prob = [0.15,0.05,0.8,0,0,0,0,0];
                end
                raction = sum(rand>=cumsum([0,prob]));
                
                switch raction
                    case 1
                       swarms(current_group).stay(completed_missions); 
                       fin = 1; 
                       past_uavs = update_table(past_uavs,swarms(current_group),fin);
                    case 2
                       emerg_land = 0;
                       fin = 1;
                       swarms(current_group).rec(emerg_land,completed_missions,fin);
                       past_uavs = update_table(past_uavs,swarms(current_group),fin);
                    case 3 
                       r_unvis = length(unvis_areas); 
                       if r_unvis == length(areas) %No areas have been visited 
                           next_pos_lead.x = 0; 
                           next_pos_lead.y = 0; 
                           next_pos_lead.id = 0;
                           while true
                               seed_pos = randi([1,r_unvis]); 
                               next_pos_lead.x = areas(unvis_areas(seed_pos)).pos.x; 
                               next_pos_lead.y = areas(unvis_areas(seed_pos)).pos.y; 
                               next_pos_lead.id = unvis_areas(seed_pos); 
                               if strcmp(areas(unvis_areas(seed_pos)).class_type,'site')
                                   continue; 
                               elseif find(vis_areas,next_pos_lead.id)
                                   continue; 
                               else
                                   vis_areas(next_pos_lead.id) = next_pos_lead.id; 
                                   unvis_areas(next_pos_lead.id) = []; 
                                   break; 
                               end
                           end
                           %free_areas = ajac_areas(next_pos_lead, areas); 
                           next_areas = voronoi_adj_areas(arrayfun(@(x) x.pos, areas), next_pos_lead,C,group_size-1);
                           next_areas = [next_pos_lead.id, next_areas];
                           current_positions.x = arrayfun( @(area) area.pos.x, areas(next_areas)); 
                           current_positions.y = arrayfun( @(area) area.pos.y, areas(next_areas));
                           current_positions.z = ones(1,group_size) * height; 
                           current_positions.id = next_areas;  

                           swarms(current_group).asc(current_positions,depots,completed_missions); 
                           past_uavs = update_table(past_uavs,swarms(current_group)); 
                           st.x = swarms(current_group).leader.current_state.x;
                           st.y = swarms(current_group).leader.current_state.y;
                           st.z = swarms(current_group).leader.current_state.z;
                       end 
                        
                end
                
            end 
        elseif strcmp(areas(clp).class_type,'area')
            
            if any(swarms(current_group).energy_levels <= swarms(current_group).leader.min_battery)
               disp("Low power capacity... Forcing descend");
               emerg_desc = 1; 
               [~, best_depot] = calculate_distance('U2I', swarms(current_group), depots, swarms.(current_group).leader); 
               swarms(current_group).desc(depots(best_depot), completed_missions); 
               past_uavs = update_table(past_uavs,swarms(current_group),0);
               next_tf = t + 1; 
               time_step = next_tf + 6; 
               emerg_land = 1; 
               swarms(current_group).rec(emerg_land,completed_missions, time_step);
               past_uavs = update_table(past_uavs, swarms(current_group),time_step); 
               flag = swarms(current_group).group_id; 
               current_group = change_id(swarms,current_group); 
               continue; 
            
            else
                 current_position = swarms(current_group).leader.current_state.area_id; 
                 r_vis = length(vis_areas); 
                 r_unvis = length(unvis_areas); 
                 if (mod(completed_missions,2)==0)
                     if (find(vis_areas, current_position))
                         if surv_areas(current_position)==0
                             prob = [0,0,0,1,0,0,0,0];
                         else
                             prob = [0,0,0,0.1,0.85,0,0,0.05];
                         end
                     elseif strcmp(past_uavs(swamrs(current_group).leader.uav_id,t-1).action, 'Move')
                         prob = [0,0,0,1,0,0,0,0];
                     elseif all(surv_areas) && r_vis == length(areas)
                         completed_missions = completed_missions + 1; 
                         surv_areas(:) = 0; 
                         vis_areas(current_position) = current_position; 
                         unvis_areas(current_position) = []; 
                         fprintf('\n---Surveillance Mission Completed---\n')
                         continue; 
                     elseif surv_areas(current_position) == 0
                         prob = [0,0,0,1,0,0,0,0]; 
                     elseif ~all(surv_areas) && r_vis == length(areas) -1
                         prob = [0,0,0,0,1,0,0,0];
                     else 
                         prob = [0,0,0,0.1,0.85,0,0,0.05];
                     end 
                     
                 elseif (mod(completed_missions,2)==1)
                     if (find(vis_areas, current_position))
                         if cov_areas(current_position) == 0
                            prob = [0,0,0,0,0,0,1,0]; 
                         else
                            prob = [0,0,0,0,0.85,0,0.1,0.05]; 
                         end 
                     elseif strcmp(past_uavs(swamrs(current_group).leader.uav_id,t-1).action, 'Move')
                         prob = [0,0,0,0,0,0,1,0];
                     elseif all(cov_areas) && r_vis == length(areas)-1
                         completed_missions = completed_missions + 1; 
                         cov_areas(:) = 0;
                         vis_areas(current_position) = current_position; 
                         unvis_areas(current_position) = [];
                         fprintf('\n---Coverage Mission Completed---\n');
                         continue;
                     elseif cov_areas(current_position) == 0
                         prob = [0,0,0,0,0,0,1,0];
                     elseif ~all(cov_areas) && r_vis == length(areas)-1
                         prob = [0,0,0,0,1,0,0,0] ; 
                     else
                         prob = [0,0,0,0,0.85,0,0.1,0.05];
                     end                     
                 end 
                 raction = sum(rand>=cumsum([0,prob])); 
                 [u2c_dists, u2u_dists] = swarms(current_group).u2u_enabled('U2U',depots,areas,cues_pos,num_cues); 
                 
                 raction = 7;
                 switch raction 
                     
                     case 4 
                        surv_areas(current_position) = 1; 
                        swarms(current_group).surv(depots, u2c_dists, completed_missions); 
                        past_uavs = update_table(past_uavs,swarms(current_group),t+1);
                        
                     case 5
                         
                     case 6
                         fprintf(" Action Hover was triggered. Warning, this action is not performed by command.\n Set possibility for action 6 to zero in all cases to avoid this behavior."); 
                         past_uavs = update_table(past_uavs,swarms(current_group),t+1);
                         
                     case 7 
                         cov_areas(current_position) = 1; 
                         swarms(current_group).cov(u2c_dists, completed_missions); 
                         past_uavs = update_table(past_uavs,swarms(current_group),t+1);

                     case 8 
                        emerg_desc = 0; 
                        [~, best_depot] = calculate_distance('U2I', swarms(current_group), depots, swarms.(current_group).leader); 
                        swarms(current_group).desc(depots(best_depot), completed_missions);
                        past_uavs = update_table(past_uavs,swarms(current_group),t+1);
                        
                     otherwise
                         error("Not an available option was given"); 
                 end 
                 
                 
                 
               
            end
        
              
        
        
        
        
        
        else 
            fprintf("\nSomething went wrong during configuration\n"); 
            break; 
        end 
        
        
        
        
        
        
        fprintf('|Action => %s |Leader => %d |Battery => %f |Time Slot => %d\n',past_uavs(swarms(current_group).leader.uav_id,t).action , swarms(current_group).leader.current_state.battery,t);
        pause(1); 
        [prev_state] = plot_update_points(swarms(current_group), colors{color_counter}, prev_state);
        t = t + 1;  
        
    end 

end

    