import numpy as np 
from program_config import * 


class DroneEnergyModel: 
    """
    This class is an offline estimation model that considers the distances 
    between the points to visit and the profile of the UAV, to compute 
    action-based energy. 
    """

    def __init__(
            self,
            v_hor:float=HORIZONTAL_VELOCITY,
            v_ver:float=VERTICAL_VELOCITY,
            alpha:float=ROTOR_AREA,
            lambda_coef:float=LAMBDA_COEF,
            dt:int=TIME_STEP_SEC[-1],
            max_battery:float=MAX_BATTERY, 
            ascent_factor:float=ASCENT_FACTOR,
            descent_factor:float=DESCENT_FACTOR[0]
    ): 

        self.g = G # Acceleration of gravity in m/s^2 
        self.p = P # Air Density in kg/m³ 
        self.alpha = alpha # Rotor disk area in m
        self.lambda_coef = lambda_coef # Coefficient for the drag profile depending on the type of UAV. 
        self.min_hover = MIN_HOVER  # Least power required to minimally hover over the ground. 
        self.dt = dt # Duration of action (10 mins per hour) in sec 
        self.max_battery = max_battery * 3600 # From Wh to J
        self.vertical_velocity = v_ver # Equivalent of 10km/h
        self.horizontal_velocity = v_hor 
        self.mass = MASS # In kg with the payload for coverage. 
        self.num_rot = NUMBER_OF_ROTORS
        self.rotor_area = self.alpha 
        self.total_rotor_area = self.num_rot * self.rotor_area

        self.ascent_factor = ascent_factor
        self.descent_factor = descent_factor

        self.bank_angle_deg = BANK_ANGLE_DEG


    def __thrust(self): 
        load_factor = 1 / np.cos(np.radians(self.bank_angle_deg))
        return self.mass * self.g * load_factor


    def __hover_power(self): 
        T = self.__thrust()
        return (T**1.5) / np.sqrt(2 * self.p * self.total_rotor_area) 


    def __profile_power(self): 
        return (self.lambda_coef * self.p * self.total_rotor_area* self.horizontal_velocity ** 3)/8


    def __horizontal_energy(self, distance_m): 
        dt_hor = distance_m / self.horizontal_velocity 
        return (self.__hover_power() + self.__profile_power()) * dt_hor


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
        # T = self.mass * self.g
        T = self.__thrust()

        # Vertical Energy in J
        E_u = self.ascent_factor * T * (self.vertical_velocity) * dt_vertical
        
        E_hover_during_vertical = self.__hover_power() * dt_vertical
        
        # Horizontal Travel in seconds hence the distance should be in meters. 
        hor_distance = self.horizontal_distance(current_node,next_node,distance_matrix)

        # --- Dorling Style --- # 
        # Hover Power (W) -> Convert to energy 
        # P_hov = np.sqrt((T)/2*self.p*self.alpha)
        # P_hov = self.__hover_power()
        #
        # # Induced drag + profile drag
        # W = (T)**2 
        # K = np.sqrt(2) * self.p * self.alpha 
        # D = np.sqrt(self.horizontal_velocity**2 + np.sqrt(self.horizontal_velocity**4 + 4 * (P_hov**4))) 
        # E_hor = (W / K) * (1 / D) * dt_hor  # J
        # --- Dorling Style --- #

        E_hor = self.__horizontal_energy(hor_distance)

        # Profile drag
        # E_r = (self.lambda_coef * self.p * self.alpha * (self.horizontal_velocity**3) * dt_hor) / 8  # J
        
        E_asc = E_hor + E_u + E_hover_during_vertical # This is in Joules

        return E_asc
        

    def hover_energy(self): 
        # P_hov = ((T**(3/2))/np.sqrt(2*self.num_rot*self.p*self.alpha)) 
        P_hov = self.__hover_power()
        E_hov = P_hov * self.dt
        
        return E_hov
        

    def coverage_energy(self, altitude, time_steps_mins): 
         # In W is the power to operate the drone as a low level base station
        time_steps_in_seconds = time_steps_mins * 60
        
        P_total = self.min_hover + (MOTOR_SPEED_MULTIPLIER*altitude/100) + P_BS # In W
        E_cov = P_total * time_steps_in_seconds
        
        return E_cov
        
    
    def descend_energy(self, current_node, next_node, altitude, distance_matrix): 
        #Total descend time in seconds 
        dt_vertical = np.abs(altitude/self.vertical_velocity) 
        # T = self.mass * self.g
        T = self.__thrust()

        # Vertical Energy in J
        E_u = self.descent_factor * T *(self.vertical_velocity)*dt_vertical 

        E_hover_during_vertical = self.__hover_power() * dt_vertical
        
        # Horizontal Travel in seconds hence the distance should be in meters. 
        horizontal_distance = self.horizontal_distance(current_node,next_node, distance_matrix)  

        # --- Dorling Style --- # 
        # Hover Power (W) -> Convert to energy 
        # P_hov = np.sqrt((T)/2*self.p*self.alpha)
        # P_hov = self.__hover_power()
        # # E_hov = P_hov * (dt_vertical + dt_hor)
        #
        # # Induced drag + profile drag
        # W = T ** 2
        # K = np.sqrt(2) * self.p * self.alpha
        # D = np.sqrt(self.horizontal_velocity**2 + np.sqrt(self.horizontal_velocity**4 + 4 * (P_hov**4)))
        # E_hor = (W / K) * (1 / D) * dt_hor  # J
        # --- Dorling Style --- # 

        E_hor = self.__horizontal_energy(horizontal_distance)
        
        # Profile drag
        E_desc = E_u + E_hor + E_hover_during_vertical # This is in Joules
        
        return  E_desc


    def move_energy(self, current_node, next_node ,distance_matrix):
        
        # Horizontal Travel in seconds hence the distance should be in meters. 
        hor_distance = self.horizontal_distance(current_node, next_node,distance_matrix)
        
        # --- Dorling Style --- #
        # T = self.mass * self.g
        # Hover Power (W) -> Convert to energy 
        # P_hov = np.sqrt((T)/2*self.p*self.alpha)
        # P_hov = self.__hover_power()
        #
        # # Induced drag + profile drag
        # W = T ** 2
        # K = np.sqrt(2) * self.p * self.alpha
        # D = np.sqrt(self.horizontal_velocity**2 + np.sqrt(self.horizontal_velocity**4 + 4 * (P_hov**4)))
        # E_hor = (W / K) * (1 / D) * dt_hor  # J
        # --- Dorling Style --- #

        E_hor = self.__horizontal_energy(hor_distance)

        # Profile drag
        E_move = E_hor

        return E_move


    def horizontal_distance(self,current_node_id,next_node_id, distance_matrix):
        return 1000 * distance_matrix[current_node_id, next_node_id] # from km to meters


