import numpy as np 


class DroneEnergyModel: 
    """
    This class is an offline estimation model that considers the distances 
    between the points to visit and the profile of the UAV, to compute 
    action-based energy. 
    """

    def __init__(self, v_hor:float=5.55, v_ver:float=2.78, alpha:float=0.2, lambda_coef:float=0.08, dt:int=600, max_battery:float=1500): 

        self.g = 9.81 # Acceleration of gravity in m/s^2 
        self.p = 1.225 # Air Density in kg/m³ 
        self.alpha = alpha # Rotor disk area in m
        self.lambda_coef = lambda_coef # Coefficient for the drag profile depending on the type of UAV. 
        self.min_hover = 30  # Least power required to minimally hover over the ground. 
        self.dt = dt # Duration of action (10 mins per hour) in sec 
        self.max_battery = max_battery * 3600 # From Wh to J
        self.vertical_velocity = 2.78 # Equivalent of 10km/h
        self.horizontal_velocity = 5.55 
        self.mass = 6.4 # In kg with the payload for coverage. 
        self.num_rot = 4 


    def stay_energy(self, energy): 
        energy += 0.0 
        return energy 


    def recover_energy(self):
        # Recharge at a fixed charging power (e.g., 200 W)
        P_charge = 350  # Watts
        energy_recovered = P_charge * self.dt  # in Joules

        return  energy_recovered


    def ascend_energy(self, current_node, next_node, altitude, distance_matrix):
        # The energy required to lift the uav from the depot to an area.
            
        #Total descend time in seconds 
        dt_vertical = altitude / self.vertical_velocity  # s
        T = self.mass * self.g

        # Vertical Energy in J
        E_u = T * (self.vertical_velocity) * dt_vertical
        
        # Horizontal Travel in seconds hence the distance should be in meters. 
        hor_distance = self.horizontal_distance(current_node,next_node,distance_matrix)
        dt_hor = hor_distance / self.horizontal_velocity 
        
        # Hover Power (W) -> Convert to energy 
        P_hov = np.sqrt((T)/2*self.p*self.alpha)
        
        # Induced drag + profile drag
        W = (T)**2 
        K = np.sqrt(2) * self.p * self.alpha 
        D = np.sqrt(self.horizontal_velocity**2 + np.sqrt(self.horizontal_velocity**4 + 4 * (P_hov**4)))
        
        E_hor = (W / K) * (1 / D) * dt_hor  # J

        # Profile drag
        E_r = (self.lambda_coef * self.p * self.alpha * (self.horizontal_velocity**3) * dt_hor) / 8  # J
        
        E_asc = E_hor + E_u + E_r # This is in Joules

        return E_asc
        

    def hover_energy(self): 
        T = self.g * self.mass 
        P_hov = ((T**(3/2))/np.sqrt(2*self.num_rot*self.p*self.alpha)) 
        E_hov = P_hov * self.dt
        
        return E_hov
        

    def coverage_energy(self, altitude, time_steps_mins): 
        P_BS = 200 # In W is the power to operate the drone as a low level base station
        time_steps_in_seconds = time_steps_mins * 60
        motor_speed_multiplier = 10.5 
        P_total = self.min_hover + (motor_speed_multiplier*altitude/100) + P_BS # In W
        E_cov = P_total * time_steps_in_seconds
        
        return E_cov
        
    
    def descend_energy(self, current_node, next_node, altitude, distance_matrix): 
        #Total descend time in seconds 
        dt_vertical = np.abs(altitude/self.vertical_velocity) 
        T = self.mass * self.g

        # Vertical Energy in J
        E_u = 0.2 * T *(self.vertical_velocity)*dt_vertical 
        
        # Horizontal Travel in seconds hence the distance should be in meters. 
        horizontal_distance = self.horizontal_distance(current_node,next_node, distance_matrix)  
        dt_hor = horizontal_distance/self.horizontal_velocity 

        
        # Hover Power (W) -> Convert to energy 
        P_hov = np.sqrt((T)/2*self.p*self.alpha)
        # E_hov = P_hov * (dt_vertical + dt_hor)

        # Induced drag + profile drag
        W = T ** 2
        K = np.sqrt(2) * self.p * self.alpha
        D = np.sqrt(self.horizontal_velocity**2 + np.sqrt(self.horizontal_velocity**4 + 4 * (P_hov**4)))
        E_hor = (W / K) * (1 / D) * dt_hor  # J
        
        # Profile drag
        E_r = (self.lambda_coef * self.p * self.alpha * (self.horizontal_velocity**3) * dt_hor) / 8  # J

        E_desc = E_u + E_r + E_hor # This is in Joules
        
        return  E_desc


    def move_energy(self, current_node, next_node ,distance_matrix):
        
        # Horizontal Travel in seconds hence the distance should be in meters. 
        hor_distance = self.horizontal_distance(current_node, next_node,distance_matrix)
        dt_hor = hor_distance / self.horizontal_velocity
        
        T = self.mass * self.g
       
        # Hover Power (W) -> Convert to energy 
        P_hov = np.sqrt((T)/2*self.p*self.alpha)
       
        # Induced drag + profile drag
        W = T ** 2
        K = np.sqrt(2) * self.p * self.alpha
        D = np.sqrt(self.horizontal_velocity**2 + np.sqrt(self.horizontal_velocity**4 + 4 * (P_hov**4)))
        E_hor = (W / K) * (1 / D) * dt_hor  # J

        # Profile drag
        E_r = (self.lambda_coef * self.p * self.alpha * (self.horizontal_velocity**3) * dt_hor) / 8  # J

        E_move = E_r + E_hor

        return E_move


    def horizontal_distance(self,current_node_id,next_node_id, distance_matrix):
        return 1000 * distance_matrix[current_node_id, next_node_id] # from km to meters


