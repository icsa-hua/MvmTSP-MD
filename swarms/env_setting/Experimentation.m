classdef  Experimentation < matlab.System
    % Untitled2 Add summary here
    %
    % This template includes the minimum set of functions required
    % to define a System object with discrete state.

    % Public, tunable properties
    properties
        num_swarms = 3; 
        sim_time = linspace(0,144,145); 
        num_cues = 10; 
        gcs_height = 25; 
        height = 100; 
        
    end

    properties(DiscreteState)
        solution
        swarm_size
        swarms
        completed_missions
        emerg_desc
        color_counter 
    end

    % Pre-computed constants
    properties(Access = private)

    end

    methods(Access = protected)
        function setupImpl(obj)
            % Create the necessary swarms. 
            % Perform one-time calculations, such as computing constants
        end

        function y = stepImpl(obj,u)
            % Implement algorithm. Calculate y as a function of input u and
            % discrete states.
            y = u;
        end

        function resetImpl(obj)
            % Initialize / reset discrete-state properties
            clear 
            clc 
            close all 
            fprintf("Swarm Simulation"); 
        end
    end
end
