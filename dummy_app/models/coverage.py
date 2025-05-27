import numpy as np 
from scipy.stats import gamma, expon, lognorm, weibull_min 
from dummy_app.tools.logger import logger 
import matplotlib.pyplot as plt


'''
Horizontal distance in these functions should be in km 
Altitudes of agent and Target also in km 
coordinates in meters 
'''

def coverage_u2c(agent_to_user_dist, agent_altitude, user_altitude, agent_pos, terrain_type='rural'):

    BW = 0.1e9 # Bandwidth in Hz (100MHz)
    NF = 6 # Noise Figure in dB 

    PU_dbm = 30 #agent transmit power # NOTE: 33 dBm or ~2W is a bit high for a medium-range UAV. Drop to 30 dBm or 1W for more realistic scenarios.
    PU_W = 10 ** ((PU_dbm - 30)/10) # Watts 

    sigma_2_dBm = -174 + NF + 10 * np.log10(BW)
    sigma_2_W = 10 ** ((sigma_2_dBm - 30) / 10) #convert from dBm (decibels referenced to 1 milliwatt) to Watts 
    
    pathloss_dB = pathloss_generation(
        agent_height=agent_altitude, 
        user_altitude=user_altitude,
        agent_user_dist=agent_to_user_dist,
        terrain_type=terrain_type,
        agent_pos=agent_pos
    )
    
    logger.info(f"Pathloss = {pathloss_dB}")

    g = 10 ** (-pathloss_dB / 10) 
    p = PU_W * g 
    sinr = p / sigma_2_W 
    R = BW * np.log2(1+sinr)

    return R, sinr



def pathloss_generation(nlos:int=1, nNlos:int=20, fc:float=2e9, c:float=(3e8/1e3), agent_height:int=1250, user_altitude:float=1.5, agent_user_dist:float=0.0, terrain_type='rural', agent_pos:tuple= ()): 

    """
    Computes the path loss (in dB) for different communication types: U2U, U2I, U2C.
    Assumes distance and heights in km.
     
    Friis free space path loss model extended to include nlos and nNlos constants for additional path loss 
    Shadowinf via gamma_probability density function 
    Fading (LoS/NLoS) probabilistically 

    alpha = Environmental constant for U2C 
    nlos = LoS additional LoS 
    nNlos = Carrier frequency in Hz 
    c = speed of light in km/s 
    """

    fc_term = 20 * np.log10(fc) + 20 * np.log10(4 * np.pi / c)

    height_difference = agent_height - user_altitude
    elevation_angle = height_difference / agent_user_dist 
    theta = np.degrees(np.arcsin(elevation_angle))

    P_los = los_probability(theta, terrain_type)
    P_nlos = 1 - P_los 

    # r = np.linalg.norm([agent_pos[0], agent_pos[1], agent_height]) / 1e7 
    r = agent_user_dist # In the case that agent position is in lat/lon coordinates, we assume agent_user_dist is already in km.
    
    fading_los = gamma_pdf(r=r,
                           agent_altitude=agent_height, 
                           fading_type='LoS',
                           terrain_type=terrain_type)
    
    fading_nlos = gamma_pdf(r=r, 
                            agent_altitude=agent_height, 
                            fading_type='NLoS', 
                            terrain_type=terrain_type)
    
    dist_term = 20 * np.log10(agent_user_dist)
    PL_los = fc_term + dist_term + nlos + fading_los 
    PL_nlos = fc_term + dist_term + nNlos + fading_nlos

    shadowing = gamma_pdf(r=r, 
                          agent_altitude=agent_height, 
                          fading_type='Shadowing_U2C', 
                          terrain_type=terrain_type)

    pathloss = P_los * PL_los + P_nlos * PL_nlos + shadowing

    return pathloss 


def gamma_pdf(r, agent_altitude:float=0.0,fading_type="LoS", terrain_type='rural'): 

    """
    Computes a fading or shadowing factor based on gamma/exponential/lognormal PDFs.

    LoS : Nakagami-m fading via Gamma 
    NLOS : Rayleigh fading via Exponential (Rayleigh is a special case of Weibull distribution)
    Shadowing U2C : Log-normal with height decay Inspired by 3GPP urban macrocell models where LoS likelihood increases with UAV height 

    Parameters:
    - xcoord, ycoord, altitude: coordinates and altitude of the UAV
    - fading_type: one of ['LoS', 'NLoS', 'Shadowing_U2U', 'Shadowing_U2I']
    - uav_height: only required for Shadowing_U2I

    Returns:
    - y: PDF value based on distance and model
    """

    if fading_type == 'LoS': 
        m = 2 if terrain_type == "forest" else 3 
        omega = 1 
        scale = omega / m 
        return gamma.pdf(r, a=m, scale=scale)
    
    elif fading_type == 'NLoS': 
        return weibull_min.pdf(r, c=1.5 if terrain_type == "forest" else 1.2)
    
    elif fading_type == 'Shadowing_U2I': 
        if agent_altitude == 0:
            raise ValueError("Agent has no altitude to be used for shadowing") 

        base_sigma = 4.2 * np.exp(-0.0046 * agent_altitude) 
        env_factor = 1.2  if terrain_type == 'forest' else 1.0 
        sigma = base_sigma * env_factor 
        return lognorm.pdf(r, s=sigma) 
    
    return 0 


def los_probability(theta, terrain_type="rural"): 
    terrain_params = {
        "urban": (29.6, 0.03),
        "forest": (10.0, 0.2),
        "mountain": (12.0, 0.1),
        "rural": (9.61, 0.16)
    }

    alpha, beta = terrain_params.get(terrain_type, (29.6, 0.03))
    return 1 / (1 + alpha * np.exp(-beta * (theta - alpha)))


def plot_pathloss_vs_distance(pathloss_func, terrain_types, comm_type='U2C', h_uav_km=1.2, h_target_km=0.0):
    distances_km = np.linspace(0.1, 5.0, 100)
    x, y, z = 1000, 1000, h_uav_km * 1000  # Dummy UAV position in meters

    plt.figure(figsize=(10, 6))
    for terrain in terrain_types:
        pl_values = [pathloss_func(d, h_uav_km, h_target_km, x, y, z, comm_type, terrain) for d in distances_km]
        plt.plot(distances_km, pl_values, label=f"{terrain}")

    plt.xlabel("Distance (km)")
    plt.ylabel("Pathloss (dB)")
    plt.title(f"Pathloss vs Distance for {comm_type}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
