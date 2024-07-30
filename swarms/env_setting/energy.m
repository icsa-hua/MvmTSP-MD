function energy = energy(action,uav,uav_map,completed_missions)

    %This function returns the energy consumed for  UAV in the case of one
    %action. It considers the physical model of the UAV, the environmental conditions
    %in a non-ideal environment, as well as the UAVs moving capabilities. 

    %-------Parameters----------%
    % - action = the selection from one of possible 8 actions. 
    % - start_uav = the starting position of the uav. (obj UAV)
    % - new_state = object holding information for the next time action of the uav. (obj UAV) 
    % - uav_map = object array holding the id and swarm position of each
    % drone.
    
    %-------Returns--------------%
    % - energy = the energy consumed by the drone for every action it
    % performs. 

    
    %%%Vertical speed 
    v_vertical = 2.78; %m/s this is the equivelant of 10km/h
    %v_horizontal = 8.33; %m/s this is the equivelant of 30 km/h      
    v_horizontal = 5.55;
    if mod(completed_missions,2)==0
        %Mass in Kg. This is the sum with the payload.
        m = 6.4; %https://www.droneblog.com/average-weights-of-common-types-of-drones/
    elseif mod(completed_missions,2)==1
        m = 6;
    end 
    g = 9.81; %Acceleration of gravity in km/s^2. 
    p = 0.2; %Density of the air in m^2.  
    a = 1.225; %Rotor disk area in Kg/m^3. 
    lambda = 0.08; %Coefficient for the drag profile depending on the type of the UAV.     
    b = 30; %Least power required to minimally hover over the ground. Minimum power to hover (doesn't says the measurement units)
    alpha = 10.5; %Motor speed multiplier also does not have measurement unit.     
    dt = 600; %Duration of every action (10 minutes of an hour) in sec
    max_energy = 1500; %Battry life in Wh.
        
    switch action 
        
        case 'Stay' %No energy consumption
            
             energy = 0; 
              
        case 'Rec' %When a uav selects recover it recharges its battery.
            
            if uav.current_state.battery == uav.max_battery
                energy = uav.current_state.battery + 0; 
            elseif uav_current_state.battery > uav.max_battery
                energy = uav.current_state.battery + 0; 
            else
                energy = (1/6)*(uav.max_battery); 
            end
            
              
        case 'Asc' %The energy required to lift the uav from a site to an area.
            
            h_k = new_state(2,uav_map.pos).height - new_state(1,uav_map.pos).height;
            dt_vertical = h_k/v_vertical;
            E_u = m*g*v_vertical*dt_vertical;
            dist = distance(new_state(2,uav_map.pos),0,3,'Hor',uav_map,new_state(1,uav_map.pos));
            dt_hor = dist/v_horizontal;
            E_hov = sqrt((m*g)/(2*p*a));
            W = (m*g)^2;
            K = sqrt(2)*p*a;
            M = 1; 
            D = sqrt(v_horizontal^2 + sqrt(v_horizontal^4 + 4*(E_hov^4)));
            E_h = (W/K)*(M/D)*dt_hor;
            dt_tot = dt_vertical + dt_hor;
            E_r = (lambda*p*a*(v_horizontal^3)*dt_hor)/8;
            E_asc = E_h + E_u + E_r; %This is in Joules
            E_asc = E_asc/dt_tot; %This is in watts,            
            T = g * m;
            num_rot = 4; 
            P = (T^(3/2)/(sqrt(2*num_rot*p*a))); %This is in W = J/s
            energy = E_asc + P;
            energy = energy * (1/6) ;
                        
        case 'Surv' %The energy to just hover on a certain altitude and surv. 
            
            P_hk = 85; %In watt this is the power consumption required to lift the UAV to height h with speed s. 
            P_cam = 0.096; %Wh Aerial Surveillance Drone: Power Management | EDGE 120 mA with 5 Volt consumption for 1/6 of an a hour.
            E_surv = (b + (alpha*new_state(1,uav_map.pos).height))*dt + P_hk*dt; %This is in Joule
            E_surv = E_surv/dt; %This is in Watts 
            energy = E_surv*(1/6) + P_cam; %This is Wh. 
              
        case 'Move' %The energy required to move the drone in different areas. 

            dist = distance(new_state(2,uav_map.pos),0,3,'Hor',uav_map,new_state(1,uav_map.pos));
            V_h = v_horizontal;             
            dt_hor = dist/V_h; 
            E_hov = sqrt((m*g)/(2*p*a));
            W = (m*g)^2;
            K = sqrt(2)*p*a;
            M = 1; 
            D = sqrt(V_h^2 + sqrt(V_h^4 + 4*(E_hov^4)));
            E_h = (W/K)*(M/D)*dt_hor;
            E_r = (lambda*p*a*(V_h^3)*dt_hor)/8;
            E_move = E_h + E_r; 
            E_move = E_move/dt_hor; 
            T = g * m;
            num_rot = 4; 
            P = (T^(3/2)/(sqrt(2*num_rot*p*a))); %This is in W = J/s
            energy = E_move + P;
            energy = energy *(1/6);
            
        case 'Hov'%The bare minimum energy to hover in place after have been place in an area. 

            T = g * m;
            num_rot = 4; 
            P = (T^(3/2)/(sqrt(2*num_rot*p*a))); %This is in W = J/s
            energy = P*(1/6); %This is Watt per hour. 
                        
        case 'Cov' %The energy to provide coverage to a cue in an area. 
            
            P_hk = 85;%In watt this is the power consumption required to lift the UAV to height h with speed s. 
            P_bs = 200; %In W the power to operate as a BS with low level functionalities. 
            E_cov = (b + (alpha*new_state(2,uav_map.pos).height))*dt + P_hk*dt + (P_bs*dt); %This is in Joule 
            E_cov = E_cov/dt; %This is in Watts.
            energy = E_cov*(1/6);
                                                  
        case 'Desc' %The energy required to land the uav from an area to a site.
    
            h_k = new_state(2,uav_map.pos).height - new_state(1,uav_map.pos).height;
            dt_vertical = abs(h_k/v_vertical);
            E_u = m*g*(-v_vertical)*dt_vertical;
            dist = distance(new_state(2,uav_map.pos),0,3,'Hor',uav_map,new_state(1,uav_map.pos));
            dt_hor = dist/v_horizontal;
            E_hov = sqrt((m*g)/(2*p*a));
            W = (m*g)^2;
            K = sqrt(2)*p*a;
            M = 1; 
            D = sqrt(v_horizontal^2 + sqrt(v_horizontal^4 + 4*(E_hov^4)));
            E_h = (W/K)*(M/D)*dt_hor;
            dt_tot = dt_vertical + dt_hor;
            E_r = (lambda*p*a*(v_horizontal^3)*dt_tot)/8;
            E_desc = E_h + E_u + E_r; %This is in Joules
            E_desc = E_desc/dt_tot; %This is in watts            
            T = g * m;
            num_rot = 4; 
            P = (T^(3/2)/(sqrt(2*num_rot*p*a))); %This is in W = J/s
            energy = E_desc + P;
            energy = energy*(1/6);
                     
    end
end

