function mvmtsp(num_agents,cost_dist, timestep, cost_energy, required_visits, vertices, depots)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    G = graph(cost_dist);     
    x_cc = arrayfun(@(ver) ver.pos.x, vertices); 
    x_cc = reshape(x_cc, length(x_cc),1);
    y_cc = arrayfun(@(ver) ver.pos.y, vertices); 
    y_cc = reshape(y_cc, length(y_cc),1);
    
    cities = arrayfun(@(city) city.contains_cues, vertices); 
    
    %Decleration of variables. 
    C = vertices(cities == 1); 
    V = length(x_cc); 
    m = num_agents;
    required_visits = randi([1,m],1,V); %r(.)
    c_ij = cost_dist; 
    e_ij = cost_energy; 
    D =  arrayfun(@(dep) dep.area_id, depots); 
    
    %Binary Variable
    x_ijk = optimvar('x_ijk', V,V, m,'Type', 'integer', 'LowerBound',0,'UpperBound',1);
    z_j = optimvar('z_j', V,m,'Type', 'integer', 'LowerBound',0,'UpperBound',1);
    u = optimvar('u',V, 'Type', 'integer', 'LowerBound', 1, 'UpperBound', V);

    %Weights for objective function 
    weight.dist = 0.55;
    weight.balance = 0.10;
    weight.energy = 0.35; 
    
    %Problem Variables / Constraints. 
    Mv_mTSP = optimproblem; 
    max_dist = optimvar('Md','LowerBound',0);
    min_dist = optimvar('md','LowerBound',0); 
    F1 = optimexpr(V,V,m); 
    F2 = optimexpr(V,V,m);
    
    
    %Graph Visualization
    figure;
    plot(G, 'XData', x_cc(:,1), 'YData', y_cc(:,1), 'NodeLabel', (1:36));
    title('Complete Undirected Graph of 36 Areas');

    
    
    %Constraints as they are written to the doc file 
    
    %1. Ensure that for each city j, the total number of times it is
        %visited matches the required visits. 
    for j = 1:V
        tmp = sum(x_ijk(:, j, :), 3);
        tmp = reshape(tmp, 1,length(tmp));
        totalVisits = sum((tmp + sum(x_ijk(j, :, :),3)),3);
        Mv_mTSP.Constraints.(['cityVisitReq_' num2str(j)]) = totalVisits <= required_visits(j);
    end 

    %2. Ensure that each edge is used as many times as necessary to achieve
        %the visit requirement for each city, consider the balance of visits,
        %so that the number of times an agent visits a city is the same as the
        %number of exits. 
    for k = 1:m 
        for j = 1:V
            inOutBalance = sum(x_ijk(:, j, k)) == sum(x_ijk(j, :, k));
            Mv_mTSP.Constraints.(['inOutBalance_' num2str(j) '_Agent_' num2str(k)]) = inOutBalance;
        end
    end
    
    %3. Ensure that city j is served by agent k, and that there must be at
        %least one outgoing or incoming edge fromto city j associated with
        %agent k
    for j = 1:V
        for k = 1:m
            Mv_mTSP.Constraints.(['ensureService_' num2str(j) '_' num2str(k)]) = z_j(j, k) <= sum(x_ijk(j, :, k)) + sum(x_ijk(:, j, k));
            Mv_mTSP.Constraints.(['activateService_' num2str(j) '_' num2str(k)]) = sum(x_ijk(j, :, k)) + sum(x_ijk(:, j, k)) >= z_j(j, k);
        end
    end 
    
    %4. Ensure that at least one agent must serve each city with cellular
    %devices. 
    for j = C  % Loop through only cities that contain cellular devices
        Mv_mTSP.Constraints.(['cityCoverage_' num2str(j.area_id)]) = sum(z_j(j.area_id, :)) >= 1;
    end
    
    %5. Salesman cannot visit the same city he is currently in
    diagonal = optimconstr(V,V,m); 
    for i = 1:V
        for k = 1:m
            diagonal(i,i,k) = x_ijk(i,i,k) == 0; 
        end 
    end 
    Mv_mTSP.Constraints.Diagonal = diagonal;
    
    %6. Subtour elimination 
    %for k = 1:m 
    %    for ii = 1:V
    %        for jj = 1:V
    %            if ii~=jj 
    %                Mv_mTSP.Constraints.(['subtourElim' num2str(ii) num2str(jj) 'Agent' num2str(k)]) = u(jj) - u(ii) + V * x_ijk(ii, jj, k) <= V - 1;
    %            end 
    %        end 
    %    end 
    %end 
    
    %7. Ensure that a depot can have multiple agents starting from its
        %positions and that an agent will have a depot. 
    for j = 1:V
       if ismember(j, D)
          for k = 1:num_agents
              depotConstraint = sum(x_ijk(:,j,k)) == 1; 
              Mv_mTSP.Constraints.(['depotOutgoing_' num2str(j) '_Agent_' num2str(k)]) = depotConstraint;
          end 
       end
    end
    
    for k = 1:m
        agentStartsFromDepot = sum(sum(x_ijk(D,:,k),1)) == 1; 
        Mv_mTSP.Constraints.(['agent_' num2str(k) '_starts_from_one_depot']) = agentStartsFromDepot;
    end 
    
    
    %8. Ensure that each depot is used by at least one agent (Optional)
    for j = D  % loop over depots only
        depotUsedByAtLeastOneAgent = sum(sum(x_ijk(:,j,:), 3)) >= 1;
        Mv_mTSP.Constraints.(['depot_' num2str(j) '_used_by_at_least_one_agent']) = depotUsedByAtLeastOneAgent;
    end
    
    %9. Ensure that each city is visited exactly the required visits
        %specified value This is the same as total visits. 
    for j = 1:V
        if ismember(j,D)
            continue; 
        end 
        cityConstraint = sum(sum(x_ijk(:, j, :), 1), 3) == required_visits(j);
        Mv_mTSP.Constraints.(['visitReqCity' num2str(j)]) = cityConstraint;
    end 
    
    
    %10. Objective Function (Minimization of energy and distance)
    for k = 1:m
        path_cost_k = sum(sum(c_ij .* x_ijk(:,:,k)));
%       F1(:,:,k) = (weight.energy.*x_ijk(:,:,k).*e_ij(:,:)) + (weight.dist.*path_cost);
        F1(:,:,k) = sum(sum(x_ijk(:,:,k).*e_ij(:,:))); 
        F2(:,:,k) = path_cost_k; 
        %Mv_mTSP.Constraints.(['MaxContraint_', num2str(k)]) = max_dist >= path_cost_k; 
        %Mv_mTSP.Constraints.(['MinContraint_', num2str(k)]) = min_dist <= path_cost_k; 
    end 
       
    F1 = sum(F1,'all'); 
    F2 = sum(F2,'all');
    %Mv_mTSP.Objective = weight.dist * F2 + weight.energy*F1 + weight.balance * (max_dist - min_dist);
    totalDistance = optimexpr; 
    for k = 1:m
        for i = 1:V
            for j = 1:V
                if i ~= j % Often not necessary, but avoids self-loops explicitly if unwanted
                    totalDistance = totalDistance + c_ij(i,j) * x_ijk(i,j,k);
                end
            end
        end
    end
    Mv_mTSP.Objective = totalDistance; 
    options = optimoptions('intlinprog','Display','final');
    [sol, fval, exitflag, output] = solve(Mv_mTSP, 'Options', options); 
    
    disp(sol); 
    edges = G.Edges(:,1); 
    
    sol.x_ijk = logical(round(sol.x_ijk)); 
    graph_sol = graph(edges(sol.x_ijk,1), edges(sol.trips,2),[], numnodes(G));
    tours = conncomp(graph_sol); 
    numtours = max(tours); 
    
    figure;
    plot(graph_sol, 'XData', x_cc(:,1), 'YData', y_cc(:,1), 'NodeLabel', (1:36));
    title('Complete Undirected Graph of 36 Areas');
    
    
   %%%%%%%
   %Extract the solution
   %%%%%%%
   
   
   startCities = zeros(1,num_agents); 
   
   for k = 1:num_agents
       for dep = depots_ids
           if any(sol.x_ijk(dep, :, k))
              incomingTrips = sum(sol.x_ijk(:,dep,k)); 
              if incomingTrips == 0
                  startCities(k) = dep; 
                  break; 
              end 
           end 
       end 
   end 
   
   paths = cell(1,num_agents); 
   
   for k = 1:num_agents 
      path = []; 
      
      if startCities(k) == 0
          continue; 
      end 
      
      current_city = startCities(k);  
      
      for step = 1:V
          next_city = find(sol.x_ijk(current_city, :, k)); 
          if isempty(next_city)
             break;  
          end
          
          path = [path, current_city]; 
          if next_city == path(1)
              break; 
          end 
          current_city = next_city; 
      end 
      paths{k} = path; 
      
   end
   
   for k = 1:numAgents
       fprintf('Path for agent %d: %s\n', k, mat2str(paths{k}));
   end
   
   figure;
   hold on;
   plot(G, 'XData', x_cc(:,1), 'YData', y_cc(:,1), 'NodeLabel', (1:V));
   colors = lines(num_agents); % Generate distinct colors for each agent
    % Plot each path
   for k = 1:num_agents
       path = paths{k};
       for i = 1:length(path)-1
           plot(x_cc(path([i, i+1]),1), y_st(path([i, i+1]),1), 'LineWidth', 2, 'Color', colors(k,:));
       end
   end
   title('Paths for all agents');
   hold off;

   
   
   
   
   
   
   
   
   
   
   
   
   

end

