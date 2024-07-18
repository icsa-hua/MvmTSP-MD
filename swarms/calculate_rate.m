function [R,sinr] = calculate_rate(comm_type,swarm, depots,areas, cues, u2u_dist, u2c_dist)


   BW = 0.1e9; 
   %BW_multi = BW/length(uav_to_x_dist);
   NF = 6;
   PU_dBm = 33; 
   PU_W = 10^(33/10 -3); 
   PU_dB = PU_dBm - 30;

   sigma_2_dBm = -174 + NF + 10*log10(BW); 
   sigma_2_W = 10^(sigma_2_dBm/10 -3); 
   sigma_2_dB = 10*log10(sigma_2_W);

   G_u2u_dB = -31.5;  
   
   if ~isempty(u2u_dist)
       u2u_dist = u2u_dist/1e3; 
   end 
   
   if ~isempty(u2c_dist)
       u2c_dist = u2c_dist/1e3; 
   end 
   
   switch comm_type 
       
       case 'U2U'

           non_leader_indices = find(~(arrayfun(@(y) y.uav_id, swarm.uavs) == swarm.leader.uav_id ));
           order = [non_leader_indices(2:end),non_leader_indices(1)]; 
           disp(order); 
           
           pl_u2u_dB = calculate_pathloss('U2U',u2u_dist,swarm,depots);
           i_u2u_dB = zeros(swarm.size,swarm.size); 
           p_u2u = zeros(swarm.size,swarm.size);
           for currentUAV = 1:swarm.size
                for otherUAV = 1:swarm.size
                    if currentUAV == otherUAV
                        p_u2u(currentUAV, otherUAV) = 0; %No self-transmission power 
                        continue;
                    end 
                    p_u2u(currentUAV, otherUAV) = PU_dB * G_u2u_dB * pl_u2u_dB(otherUAV, currentUAV);
                    %p_u2u_dB(currentUAV, otherUAV) = PU_dBm + G_u2u_dB - pl_u2u_dB(currentUAV, otherUAV);
                end 
                
                % Calculate the interference power from other UAVs to this non-leader UAV
                for uu = order(order ~= currentUAV)
                    if uu ~= currentUAV
                       i_u2u_dB(currentUAV,uu) = PU_dB * G_u2u_dB * pl_u2u_dB(uu,currentUAV);
                       %i_u2u_dB(currentUAV,uu) = PU_dBm + G_u2u_dB - pl_u2u_dB(otherUAV, currentUAV);
                    end 
                end

           end
           
           %i_u2u_dB(i_u2u_dB==0) = [];
           total_i_u2u_dB = sum(i_u2u_dB, 1);
           total_p_u2u = sum(p_u2u,1); 
           
           total_i_u2c_dB = zeros(1,swarm.size);
           [~, num_cues] = size(cues.pos);
           num_cues = num_cues/2;
           pl_u2c_dB = calculate_pathloss('U2C',u2c_dist,swarm,[]); 
           
           for ii = 1:swarm.size
               T = randi([0,1],1,num_cues); 
               g_u2c = 0; 
               for jj = 1:num_cues
                   g_u2c = g_u2c + 10^(-pl_u2c_dB(ii,jj)/10);
                   i_cue = PU_W * g_u2c; 
                   i_cue = 10 * log10(i_cue);
                   total_i_u2c_dB(1,ii) = total_i_u2c_dB(1,ii) + T(jj) * i_cue;
               end
           end 
           
           sinr_dB = zeros(1,swarm.size);
           R_dB = zeros(size(sinr_dB)); 
           for ii = 1:swarm.size
              sinr_dB(ii) = total_p_u2u(ii) / (sigma_2_dB + total_i_u2c_dB(ii) + total_i_u2u_dB(ii));
              R_dB(ii) = BW/2*log2(1+sinr_dB(ii));
           end
           
           sinr = sinr_dB; 
           R = R_dB/1e6;
           
       case 'U2I' 
           
           [u2i_dist, best_depot] = calculate_distance('U2I',swarm, depots,[]);
           if ~isempty(u2i_dist)
               u2i_dist = u2i_dist/1e3; 
           end
           
           pl_u2i_dB = calculate_pathloss('U2I', u2i_dist, swarm, best_depot);
           rho = ~eye(swarm.size); 
           sinr_dB = zeros(size(pl_u2i_dB)); 
           R_dB = zeros(size(sinr_dB)); 
           
           %Added the CUE interference to the U2I signals. 
           total_i_u2c_dB = zeros(1,swarm.size);
           [~, num_cues] = size(u2c_dist);
           pl_u2c_dB = calculate_pathloss('U2C',u2c_dist,swarm,[]); 
           
           for ii = 1:swarm.size
               T = randi([0,1],1,num_cues); 
               g_u2c = 0; 
               for jj = 1:num_cues
                   g_u2c = g_u2c + 10^(-pl_u2c_dB(ii,jj)/10);
                   i_cue = PU_W * g_u2c; 
                   i_cue = 10 * log10(i_cue);
                   total_i_u2c_dB(1,ii) = total_i_u2c_dB(1,ii) + T(jj) * i_cue;
               end
           end 
           %-----------------------------------------------
           
           for ii = 1:swarm.size
             gain =  10^(-pl_u2i_dB(ii)/10);
             p_tx_gcs_dB = PU_W * gain;
             p_tx_gcs_dB = 10*log10(p_tx_gcs_dB);
             i_u2i_dB = 0;
             for jj = 1:swarm.size
                 i_u2i_dB = i_u2i_dB + rho(ii,jj) * p_tx_gcs_dB; 
             end 
             sinr_dB(ii) = p_tx_gcs_dB/(sigma_2_dB + total_i_u2c_dB(ii) + i_u2i_dB); 
             R_dB(ii) = BW*log2(1+sinr_dB(ii)); 
           end 
           
           sinr = sinr_dB; 
           R = R_dB/1e6; 
   
       case 'U2C' 
           pl_u2c_dB = calculate_pathloss('U2C',u2c_dist, swarm,[]); 
           [~, num_cues] = size(u2c_dist);
           sinr = zeros(1,swarm.size); 
           rate = zeros(1,swarm.size); 
           for uav = 1:swarm.size
               p_cue_dB = zeros(1,num_cues); 
               cue_sinr = zeros(1,num_cues); 
               cue_rate = zeros(1,num_cues); 
               for cue = 1:num_cues
                   gain = 10^(-pl_u2c_dB(uav,cue)/10); 
                   p_cue_dB(1,cue) = PU_W * gain;
                   p_cue_dB(1,cue) = 10*log10(p_cue_dB(1,cue));
                   cue_sinr(1,cue) = p_cue_dB(1,cue)/sigma_2_dB; 
                   cue_rate(1,cue) = (BW)*log2(1+cue_sinr(1,cue)); 
               end 
               sinr(uav) = sum(cue_sinr(:))/num_cues; 
               rate(uav) = sum(cue_rate(:))/num_cues; 
               R = rate/1e6; 
           end 
       
       
       
       
       case 'relay'
           
       
   end 
    

















end

