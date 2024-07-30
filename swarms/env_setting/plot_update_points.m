function prev_state = plot_update_points(swarm, colors, prev_state)


    color = cell2mat(colors);
    theta = 42.44; 
    
    leader = swarm.leader; 
    
    x_coords = arrayfun(@(uav) uav.next_state.x , swarm.uavs); 
    y_coords = arrayfun(@(uav) uav.next_state.y , swarm.uavs);
    z_coords = arrayfun(@(uav) uav.next_state.z , swarm.uavs);
    
    h2 = plot3(x_coords, y_coords, z_coords,'hexagram','Color',color,'MarkerSize',15,'LineWidth',3);
    h = plot3(leader.next_state.x, leader.next_state.y, leader.next_state.z, 'hexagram','Color','r','MarkerSize',20,'LineWidth',3);

    action = leader.next_state.action; 
    switch action
        
        case 'Cov'
            cells = zeros(swarm.size,2); 
            for uav = 1:swarm.size
                radius = swarm.uavs(uav).next_state.z * tand(theta); 
                ti = linspace(0,2*pi,100);
                x_cord = swarm.uavs(uav).next_state.x + radius*cos(ti);
                y_cord = swarm.uavs(uav).next_state.y + radius*sin(ti); 
                cells(uav,1) = x_cord; 
                cells(uav,2) = y_cord; 
            end 
            ksi = zeros(100); 
            h3 = plot3(cells(:,1), cells(:,2),ksi,'--g'); 
            
        case 'Surv' 
            base = zeros(size(z_coords));
            points = [x_coords, y_coords, base; x_coords, y_coords, z_coords]; 
            h3 = plot3(points(:,1), points(:,2), points(:,3),'-.b') ;            
        
        case 'Move' 
            
        otherwise 
            return; 

    end 
    
    if leader.next_state.time > 1
            delete(prev_state)
    end

    prev_state.h = h; 
    prev_state.h2 = h2; 
    prev_state.h3 = h3; 
    
    drawnow; 
    



end

