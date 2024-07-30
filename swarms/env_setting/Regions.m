classdef Regions
    %UNTITLED5 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        area_id
        class_type
        pos 
        contains_cues 
        gcs_height
    end
    
    methods
        function obj = Regions(area_id, class_type, pos, contains_cue,gcs_height)
            %UNTITLED5 Construct an instance of this class
            %   Detailed explanation goes here
            obj.area_id = area_id; 
            obj.class_type = class_type; 
            obj.pos.x = pos.x;
            obj.pos.y = pos.y; 
            obj.contains_cues = contains_cue; 
            obj.gcs_height = gcs_height;
        end
        
    end
end

