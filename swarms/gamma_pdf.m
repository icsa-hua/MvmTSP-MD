function y = gamma_pdf(type,uav, uav_height_diff)

    % Extract the UAV's current position in the gamma structure
    gamma.x = uav.current_state.x; 
    gamma.y = uav.current_state.y; 
    gamma.z = uav.current_state.z; 

    % Form a point vector from the UAV's current position
    point = [gamma.x, gamma.y, gamma.z];

    % Calculate the norm of the point vector and scale it
    % The scaling factor needs to be clarified or adjusted according to the specific use case
    r = sqrt(sum(point.^2, 2)) / 1e7;

    % Switch on the type of propagation model
    switch type
        case 'LoS'
            % Parameters for the Gamma distribution
            m = 3; % Shape parameter (also known as k or alpha)
            omega = 1; % Scale multiplier
            shape_param = m; 
            scale_param = omega / m; % Scale parameter (theta)

            % Create the Gamma distribution with specified parameters
            mypdf = makedist('Gamma', 'a', shape_param, 'b', scale_param);

            % Evaluate the PDF at distances r
            y = pdf(mypdf, r);
    
        case 'NLoS'
           %This is the equivellant of Rayleigh Fading using the
           %exponential distribution. (exp(1)).
           pd = makedist('Exponential','mu',1);
           y  = pdf(pd,r);
           
       case 'Shadowing_U2U'
           %Log normal distribution for shadowing in U2U communication. 
           %Not negligible. 
           std = 3; %dB
           pd = makedist('Lognormal','sigma',std);
           y = pdf(pd,r);
           
       case 'Shadowing_U2I'
           %Log normal distribution for shadowing in U2I communciation.
           std = 4.2*exp(-0.0046*uav_height_diff); 
           pd = makedist('Lognormal','sigma',std);
           y =  pdf(pd,r);
       
       case 'Shadowing_U2C'
           %Log normal distribution for shadowing in U2I communciation.
           std = 4.2*exp(-0.0046*uav_height_diff); 
           pd = makedist('Lognormal','sigma',std);
           y =  pdf(pd,r);
           
       otherwise 
           y = 0; 
    end

end

