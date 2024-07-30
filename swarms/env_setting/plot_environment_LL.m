function plot_environment_LL(Y,uu,vv,option,H)
    disp("plot_environment");
    %This function creates the environment for plotting. 
    
    %-------Parameters----------%
    % - X = the whole set of places.  
    % - uu = x_coords for generated cue in the sites which are random
    % - vv = y_coords for generated cue in the sites which are random
    % - CUE = the coordinates for the cue in the areas.  
    
    %-------Returns--------------%
    % Nothing.
    clf        
    figure(1)
    hold on 
    %Plot the layout of the environment on the ground seperating the areas
    %and the sites. 

    conn = [Y(8).pos.x, Y(8).pos.y, 0; Y(8).pos.x, Y(8).pos.y, H];
    conn2 = [Y(9).pos.x, Y(9).pos.y, 0; Y(9).pos.x, Y(9).pos.y, H];
    
    m = zeros(length(uu),1); 
    m(:) = 1.5;     
    
    markers = ['o', 's', '^', 'd', 'p']; 
    markerIdx = randi(length(markers), size(uu));    
    uu_Size = size(uu); 
    plot3(conn(:,1),conn(:,2),conn(:,3),'Marker','diamond','LineStyle', '--','Color',[0.6350 0.0780 0.1840]);  
    plot3(conn2(:,1),conn2(:,2),conn2(:,3),'Marker','diamond','LineStyle', '--','Color',[0.6350 0.0780 0.1840]);          
    %plot3(uu,vv,m,'pentagram','Color',color1,'MarkerSize',10);
    for i = 1:uu_Size(1)
        for jj = 1: uu_Size(2)
            plot3(uu(i,jj), vv(i,jj), m(jj), markers(markerIdx(jj)), 'MarkerSize', 8, 'MarkerFaceColor', 'auto');
        end
    end
    colormap(parula(length(markers)));
    colorbar; 
    areas_xx = arrayfun(@(y) y.pos.x, Y); 
    areas_yy = arrayfun(@(y) y.pos.y, Y);
    voronoi(areas_xx,areas_yy);

    n = zeros(length(Y),1); 
    X2 = [Y(8).pos.x, Y(8).pos.y; Y(9).pos.x, Y(9).pos.y;];
    nump = length(Y); nump2 = size(X2,1);
    plabels = arrayfun(@(n) {sprintf('z%d', n)}, (1:nump)');
    plabels2 = arrayfun(@(n) {sprintf('s%d', n)}, (1:nump2)');
           
    switch option 
        
        case 'Sim_area' 
            xlabel('Longitude (m)');
            ylabel('Latitude (m)');
            zlabel('Height (m)'); 
            Hpl = text(Y(:,1)+0, Y(:,2)+250, plabels, 'color', 'r','FontWeight', 'bold', 'HorizontalAlignment','center', 'BackgroundColor', 'none');
            Hp2 = text(X2(:,1)+250, X2(:,2)+0, plabels2, 'color', 'b','FontWeight', 'bold', 'HorizontalAlignment','center', 'BackgroundColor', 'none');
    
        case 'Small_area' 
            xlabel('Longitude (m)');
            ylabel('Latitude (m)');
            Hpl = text(Y(:,1)+50, Y(:,2)+100, plabels, 'color', 'r','FontWeight', 'bold', 'HorizontalAlignment','center', 'BackgroundColor', 'none');
            Hp2 = text(X2(:,1)+50, X2(:,2)+100, plabels2, 'color', 'b','FontWeight', 'bold', 'HorizontalAlignment','center', 'BackgroundColor', 'none');
    
        case 'Real_area_20' 
            xlabel('Longitude (m)');
            ylabel('Latitude (m)');
            zlabel('Height (m)'); 
            Hpl = text(Y(:,1)+0, Y(:,2)+100, plabels, 'color', 'r','FontWeight', 'bold', 'HorizontalAlignment','center', 'BackgroundColor', 'none');
            Hp2 = text(X2(:,1)+50, X2(:,2)+0, plabels2, 'color', 'b','FontWeight', 'bold', 'HorizontalAlignment','center', 'BackgroundColor', 'none');
        
        case 'Real_area_40' 
            xlabel('Longitude (m)');
            ylabel('Latitude (m)');
            zlabel('Height (m)'); 
            Hpl = text(areas_xx+0, areas_yy+100, plabels, 'color', 'r','FontWeight', 'bold', 'HorizontalAlignment','center', 'BackgroundColor', 'none');
            Hp2 = text(X2(:,1)+50, X2(:,2)+0, plabels2, 'color', 'b','FontWeight', 'bold', 'HorizontalAlignment','center', 'BackgroundColor', 'none');
        
        case 'other' 
            xlabel('Longitude (m)');
            ylabel('Latitude (m)');
            zlabel('Height (m)'); 
            Hpl = text(Y(:,1)+50, Y(:,2)+100, plabels, 'color', 'r','FontWeight', 'bold', 'HorizontalAlignment','center', 'BackgroundColor', 'none');
            Hp2 = text(X2(:,1)+50, X2(:,2)+100, plabels2, 'color', 'b','FontWeight', 'bold', 'HorizontalAlignment','center', 'BackgroundColor', 'none');
    
    end 
    

    zlabel('Altitude (m)');   
    title('Swarm: Actions and Topology');
    plot3(areas_xx,areas_yy,n,'sk');
    plot3(Y(8).pos.x,Y(8).pos.y,0,'or',Y(9).pos.x,Y(9).pos.y,0,'or');
    legend('GCS_1','GCS_2','CUEs');

    %hold off;    
    %clear -except radius X h theta new_state t group_size leader    
end


