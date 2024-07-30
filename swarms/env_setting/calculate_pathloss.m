function PL = calculate_pathloss(comm_type, u2x_distance,swarm, best_depot)

    %Consider all U2U pairs as LoS
    
    %alpha = 10.39; %Environmental Constant for U2I channel model 
    %beta = 0.05; %Environmental Constant for U2I channel model 
    alpha = 29.6;
    beta = 0.03; 
    nlos = 1; %Additional attenuation factor due to the LoS connections.  (8)
    nNlos = 20; %Additional attenuation factor due to the nLos connections. 
    fc = 2e9; %Carrier Frequency in Hertz. 
    c = 3e8/1e3; %Light speed in km/s. Since distance is in Km. 

    switch comm_type
        case 'U2U'
            PL = zeros(swarm.size, swarm.size);
            for ii = 1:swarm.size
                for jj = 1:swarm.size
                   
                    shadowing = gamma_pdf('Shadowing_U2U',swarm.uavs(ii),0);
                    Naka_fading = gamma_pdf('LoS',swarm.uavs(ii));
                    if u2x_distance(ii,jj) <= 0
                        PL(ii,jj) = 0; 
                        continue; 
                    end 
                    PL(ii,jj) = 20*log10(u2x_distance(ii,jj)) + 20*log10(fc) + 20*log10(4*pi/c) + nlos; 
                    PL(ii,jj) = PL(ii,jj) + shadowing + Naka_fading; 
                end 
            end
            
        case 'U2I'
            gcs_height = 25/1e3; 
            PL = zeros(1,swarm.size); 
            for uav = 1:swarm.size
                height_diff = (swarm.uavs(uav).current_state.z/1e3) - gcs_height; 
                theta = asind(height_diff/u2x_distance(uav,best_depot)); 
                shadowing = gamma_pdf('Shadowing_U2I',swarm.uavs(uav), height_diff); 
                P_los = 1/(1 + alpha*exp(-beta*(theta - alpha)));
                P_nlos = 1 - P_los;
                PL_los = 20*log10(fc) + 20*log10(4*pi/c) + 20*log10(u2x_distance(uav,best_depot)) + nlos;
                Naka_fading = gamma_pdf('Los', swarm.uavs(uav), height_diff); 
                PL_los = PL_los  + Naka_fading; 
                PL_nlos= 20*log10(fc) + 20*log10(4*pi/c) + 20*log10(u2x_distance(uav,best_depot)) + nNlos;
                Rayl_fading = gamma_pdf('NLoS',swarm.uavs(uav), height_diff); 
                PL_nlos = PL_nlos + Rayl_fading;
                PL(uav) =  P_los * PL_los + P_nlos*PL_nlos + shadowing;
            end 
            
        case 'U2C'
            [~, num_cues] = size(u2x_distance); 
            PL = zeros(swarm.size,num_cues); 
            for uav = 1:swarm.size 
                cue_height = 1.5; 
                height_diff = swarm.uavs(uav).current_state.z - cue_height;
                height_diff = height_diff/1e3;
                
                for cue = 1:num_cues
                    if (u2x_distance(uav,cue) == 0) 
                        continue;
                    end
                    elevation_angle = (height_diff)/u2x_distance(uav,cue); 
                    theta = asind(elevation_angle); 
                    shadowing = gamma_pdf('Shadowing_U2C',swarm.uavs(uav),height_diff); 
                    P_los = 1/(1 + alpha*exp(-beta*(theta-alpha)));
                    P_nlos = 1 - P_los; 
                    Naka_fading = gamma_pdf('Los', swarm.uavs(uav),height_diff);
                    PL_los = 20*log10(fc) + 20*log10(4*pi/c) + 20*log10(u2x_distance(uav,cue)) + nlos;
                    Rayl_fading = gamma_pdf('NLoS',swarm.uavs(uav),height_diff);
                    PL_nlos = 20*log10(fc) + 20*log10(4*pi/c) + 20*log10(u2x_distance(uav,cue)) + nNlos;
                    PL_los = PL_los + Naka_fading;
                    PL_nlos = PL_nlos + Rayl_fading;
                    PL_uav_cue = P_los*PL_los + P_nlos*PL_nlos + shadowing;
                    PL(uav,cue) = PL_uav_cue; 
                end 
            end
    end
end

