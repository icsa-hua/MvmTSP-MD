function testmvmtsp(vertices, cost_dist, depots)

    
n = 36; % Number of cities, including depots
m = 5;  % Number of agents
r = randi([1 2], n, 1); % Random minimum visits for each city
C = rand(n); % Random symmetric cost matrix
C = (C + C') / 2; % Making the cost matrix symmetric

% Assuming D contains the first few indices of V
numDepots = 3;
D = 1:numDepots;
V = 1:n;
A = 1:m;

% Decision variables
x = optimvar('x', n, n, m, 'Type', 'integer', 'LowerBound', 0, 'UpperBound', 1);
y = optimvar('y', n, m, 'Type', 'integer', 'LowerBound', 0, 'UpperBound', 1);

prob = optimproblem;

% Constraint 1: Depot has only outgoing edges
for k = A
    for d = D
        prob.Constraints.(['depotOut_' num2str(d) '_' num2str(k)]) = sum(x(d,:,k), 'all') == 1;
    end
end

% Constraint 3: Each city can be visited only once by each agent
for k = A
    for i = V
        if ~ismember(i, D)
            prob.Constraints.(['visitOnce_' num2str(i) '_' num2str(k)]) = sum(x(:,i,k), 'all') + sum(x(i,:,k), 'all') == 2 * y(i,k);
        end
    end
end

% Constraint 4: Minimum Visits
for i = V
    prob.Constraints.(['minVisits_' num2str(i)]) = sum(y(i,:), 'all') == r(i);
end

% Constraint 6: Single depot per agent
for k = A
    prob.Constraints.(['singleDepot_' num2str(k)]) = sum(y(D,k), 'all') == 1;
end

% Constraint 7: Agent visits all non-depot cities
for k = A
    prob.Constraints.(['visitAllCities_' num2str(k)]) = sum(y(setdiff(V, D), k), 'all') == numel(setdiff(V, D));
end

% Constraint 8: Return to Depot
for k = A
    for d = D
        prob.Constraints.(['returnDepot_' num2str(d) '_' num2str(k)]) = sum(x(:,d,k), 'all') == 1;
    end
end


objExpr = 0;
for k = A
    for i = V
        for j = V
            objExpr = objExpr + C(i,j) * x(i,j,k);
        end
    end
end
prob.Objective = objExpr;

% Options and solve
options = optimoptions('intlinprog', 'Display', 'iter');
[sol, fval, exitflag, output] = solve(prob, 'Options', options);

% Display results
disp(sol.x);
disp(fval);





    G = graph(cost_dist);
    x_cc = arrayfun(@(ver) ver.pos.x, vertices); 
    x_cc = reshape(x_cc, length(x_cc),1);
    y_cc = arrayfun(@(ver) ver.pos.y, vertices); 
    y_cc = reshape(y_cc, length(y_cc),1);

    cities = arrayfun(@(city) city.contains_cues, vertices); 
 
    V = [x_cc,y_cc]; 
    v = length(V);
    edges = G.Edges(:,1); 
    D = arrayfun(@(d) d.pos, depots); 
    depots_ids = arrayfun(@(d) d.area_id, depots); 
    A = 1:4; 
    m = length(A);
    r = randi([1 m],1,length(V)); 
    c_ij = cost_dist; 
    

    x_ijk = optimvar("x_ijk",v,v,m,'Type','integer','LowerBound',0,'UpperBound',1);
    y_ik = optimvar("y_ik",v,m,'Type','integer','LowerBound',0,'UpperBound',1); 
    
    mvmtsp = optimproblem; 
    totalDistance = optimexpr; 
    for k = 1:m
        for i = 1:length(V)
            for j = 1:length(V)
                if i ~= j % Often not necessary, but avoids self-loops explicitly if unwanted
                    totalDistance = totalDistance + c_ij(i,j) * x_ijk(i,j,k);
                end
            end
        end
    end
    mvmtsp.Objective = totalDistance; 

    
    %Constraint 1. 
    for k = 1:m 
        for d = depots_ids
            mvmtsp.Constraints.(['depotOut_' num2str(d) '_' num2str(k)]) = sum(x_ijk(d,:,k), 'all') == 1;
        end 
    end 
    
    %Constraint 2 
    for k = A
        for i = 1:length(V)
            if ismember(i,depots_ids)
                mvmtsp.Constraints.(['visitOnce_' num2str(i) '_' num2str(k)]) = sum(x_ijk(:,i,k), 'all') + sum(x_ijk(i,:,k), 'all') == 2 * y_ik(i,k);
            end
        end
    end
    
    for i = 1:length(V)
        mvmtsp.Constraints.(['minVisits_' num2str(i)]) = sum(y_ik(i,:), 'all') >= r(i);
    end
    
    % Constraint 6: Single depot per agent
    for k = A
        mvmtsp.Constraints.(['singleDepot_' num2str(k)]) = sum(y_ik(depots_ids,k), 'all') == 1;
    end
    
    
    % Constraint 7: Agent visits all non-depot cities
    z = 1:v; 
    for k = A
        mvmtsp.Constraints.(['visitAllCities_' num2str(k)]) = sum(y_ik(setdiff(z, depots_ids), k), 'all') == numel(setdiff(z, depots_ids));
    end

    % Constraint 8: Return to Depot
    for k = A
        for d = depots_ids
            mvmtsp.Constraints.(['returnDepot_' num2str(d) '_' num2str(k)]) = sum(x_ijk(:,d,k), 'all') == 1;
        end
    end
    
    objExpr = 0;
    for k = A
        for i = 1:v
            for j = 1:v
                objExpr = objExpr + c_ij(i,j) * x_ijk(i,j,k);
            end
        end
    end
    mvmtsp.Objective = objExpr;

    % Options and solve
    options = optimoptions('intlinprog', 'Display', 'iter');
    [sol, fval, exitflag, output] = solve(mvmtsp, 'Options', options);

    % Display results
    disp(sol.x_ijk);
    disp(fval);
    
    

end

