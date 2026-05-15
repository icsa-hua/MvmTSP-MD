"""
This script provides all the required functions and methodology to 
calculate the achievable data rate for each agent. It imitates the 
environmental conditions and the probability a  link to be LoS, or NLoS.
"""

from dummy_app.tools.logger import logger 

import os 
import uuid 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from dataclasses import asdict, dataclass
from scipy.stats import norm 
from scipy.integrate import quad
from scipy.stats import gamma, lognorm, weibull_min 


'''
Horizontal distance in these functions should be in km 
Altitudes of agent and Target also in km 
coordinates in meters 
'''


@dataclass
class CoverageDiagnostics:
    terrain_type: str
    bandwidth_hz: float
    frequency_hz: float
    shadowing_db: float
    interference_w: float
    agent_altitude_km: float
    user_altitude_km: float
    horizontal_distance_km: float
    height_difference_km: float
    theta_deg: float
    fspl_db: float
    p_los: float
    p_nlos: float
    los_loss_db: float
    nlos_loss_db: float
    final_pathloss_db: float
    sample_count: int = 1

    def to_dict(self):
        return asdict(self)


def average_coverage_diagnostics(records):
    records = list(records)
    if not records:
        return None

    terrain_values = {record.terrain_type for record in records}
    terrain_type = records[0].terrain_type if len(terrain_values) == 1 else "mixed"

    def avg(field_name):
        return float(np.mean([getattr(record, field_name) for record in records]))

    return CoverageDiagnostics(
        terrain_type=terrain_type,
        bandwidth_hz=avg("bandwidth_hz"),
        frequency_hz=avg("frequency_hz"),
        shadowing_db=avg("shadowing_db"),
        interference_w=avg("interference_w"),
        agent_altitude_km=avg("agent_altitude_km"),
        user_altitude_km=avg("user_altitude_km"),
        horizontal_distance_km=avg("horizontal_distance_km"),
        height_difference_km=avg("height_difference_km"),
        theta_deg=avg("theta_deg"),
        fspl_db=avg("fspl_db"),
        p_los=avg("p_los"),
        p_nlos=avg("p_nlos"),
        los_loss_db=avg("los_loss_db"),
        nlos_loss_db=avg("nlos_loss_db"),
        final_pathloss_db=avg("final_pathloss_db"),
        sample_count=sum(int(record.sample_count) for record in records),
    )

def coverage_u2c(
        agent_to_user_dist_km,
        agent_altitude_km,
        user_altitude_km,
        terrain_type='rural', 
        bandwidth_hz=100e6, # Bandwidth in Hz (100MHz)
        noise_figure_db=6, 
        tx_power_dbm=30, 
        interference_W=0.0,
        carrier_frequency_hz=2.4e9,
        shadowing_db=None,
        return_details=False,
    ):
    # NOTE: 33 dBm or ~2W is a bit high for a medium-range UAV. Drop to 30 dBm or 1W for more realistic scenarios.
    tx_power_W = 10 ** ((tx_power_dbm - 30)/10) # Watts 

    noise_power_dbm = -174 + noise_figure_db + 10 * np.log10(bandwidth_hz)

    noise_power_W = 10 ** ((noise_power_dbm - 30) / 10) #convert from dBm (decibels referenced to 1 milliwatt) to Watts 
    
    pathloss_result = pathloss_generation(
        agent_height_km=agent_altitude_km, 
        user_altitude_km=user_altitude_km,
        agent_user_dist_km=agent_to_user_dist_km,
        terrain_type=terrain_type,
        fc=carrier_frequency_hz,
        shadowing_db=shadowing_db,
        bandwidth_hz=bandwidth_hz,
        interference_W=interference_W,
        return_details=return_details,
    )

    if return_details:
        pathloss_dB, diagnostics = pathloss_result
    else:
        pathloss_dB = pathloss_result
        diagnostics = None
    
    channel_gain = 10 ** (-pathloss_dB / 10) 
    received_power_W = tx_power_W * channel_gain 
    sinr = received_power_W / max(noise_power_W + interference_W, 1e-12)
    rate_bps = bandwidth_hz * np.log2(1+sinr)

    if return_details:
        return rate_bps, sinr, diagnostics
    return rate_bps, sinr



def pathloss_generation(
        eta_los_db:int=1,
        eta_nlos_db:int=20,
        fc:float=2.4e9,
        c:float=3e8,
        agent_height_km:float=1.25,
        user_altitude_km:float=0.0015,
        agent_user_dist_km:float=0.001,
        terrain_type: str ='rural', 
        shadowing_db = None,
        bandwidth_hz: float = 100e6,
        interference_W: float = 0.0,
        return_details: bool = False,
    ): 

    """
    Computes the path loss (in dB) for different communication types: U2C.
    Assumes distance and heights in km.
     
    Friis free space path loss model extended to include nlos and nNlos constants for additional path loss 
    Shadowinf via gamma_probability density function 
    Fading (LoS/NLoS) probabilistically 

    alpha = Environmental constant for U2C 
    nlos = LoS additional LoS 
    nNlos = Carrier frequency in Hz 
    c = speed of light in km/s 
    """

    
    d_km = max(agent_user_dist_km, 1e-6)
    h_diff_km = agent_height_km - user_altitude_km

    d_3d_m = np.sqrt((d_km * 1000) ** 2 + (h_diff_km * 1000) **2)

    theta = np.degrees(np.arctan2(h_diff_km,d_km))

    prob_los = los_probability(theta, terrain_type)
    prob_nlos = 1 - prob_los 

    # In the case that agent position is in lat/lon coordinates, we assume agent_user_dist is already in km.
    
    fading_los = gamma_pdf(r=agent_user_dist_km,
                           agent_altitude=agent_height_km, 
                           fading_type='LoS',
                           terrain_type=terrain_type)
    
    fading_nlos = gamma_pdf(r=agent_user_dist_km, 
                            agent_altitude=agent_height_km, 
                            fading_type='NLoS', 
                            terrain_type=terrain_type)
    
    fspl_db = 20 * np.log10(4 * np.pi *fc * d_3d_m / c)
    
    loss_los_db = fspl_db + eta_los_db + fading_los 
    loss_nlos_db = fspl_db + eta_nlos_db + fading_nlos

    if shadowing_db is None:
        shadowing_db = gamma_pdf(
            r=agent_user_dist_km,
            agent_altitude=agent_height_km,
            fading_type='Shadowing_U2C',
            terrain_type=terrain_type,
        )

    pathloss = prob_los * loss_los_db + prob_nlos * loss_nlos_db + shadowing_db

    diagnostics = CoverageDiagnostics(
        terrain_type=terrain_type,
        bandwidth_hz=float(bandwidth_hz),
        frequency_hz=float(fc),
        shadowing_db=float(shadowing_db),
        interference_w=float(interference_W),
        agent_altitude_km=float(agent_height_km),
        user_altitude_km=float(user_altitude_km),
        horizontal_distance_km=float(d_km),
        height_difference_km=float(h_diff_km),
        theta_deg=float(theta),
        fspl_db=float(fspl_db),
        p_los=float(prob_los),
        p_nlos=float(prob_nlos),
        los_loss_db=float(loss_los_db),
        nlos_loss_db=float(loss_nlos_db),
        final_pathloss_db=float(pathloss),
    )

    if return_details:
        return pathloss, diagnostics
    return pathloss 


def gamma_pdf(r, agent_altitude:float=0.0,fading_type="LoS", terrain_type='rural'): 

    """
    Computes a fading or shadowing factor based on gamma/exponential/lognormal PDFs.
    LoS : Nakagami-m fading via Gamma 
    NLOS : Rayleigh fading via Exponential (Rayleigh is a special case of Weibull distribution)
    Shadowing U2C : Log-normal with height decay Inspired by 3GPP urban macrocell models where LoS likelihood increases with UAV height 
    """

    if fading_type == 'LoS': 
        m = 2 if terrain_type == "forest" else 3 
        omega = 1 
        scale = omega / m 
        return gamma.pdf(r, a=m, scale=scale)
    
    elif fading_type == 'NLoS': 
        return weibull_min.pdf(r, c=1.5 if terrain_type == "forest" else 1.2)
    
    elif fading_type in {'Shadowing_U2I', 'Shadowing_U2C'}: 

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


def coverage_probability(
    cluster,
    num_users,
    savefile_name=None,
    directory=None,
    snr=None,
    lambda_var:float=1,
    seed:int=42,
    save_artifacts: bool = False,
):

    def rayleigh_pdf_ppp(x): 
        return 2 * lambda_var * np.pi * x * np.exp(-lambda_var * np.pi * x ** 2) 

    theta_snr_db = np.linspace(-1,30,20) 

    cov_probability = np.zeros((len(snr), len(theta_snr_db)))

    for i, snr_db in enumerate(snr): 
        for j, threshold in enumerate(theta_snr_db): 
            prob = 1 - norm.cdf(threshold, loc=snr_db, scale=10) 
            fun = lambda r: prob * rayleigh_pdf_ppp(r) 
            cov_probability[i, j],_ = quad(fun, 0, np.inf)

    coverage_PR = np.mean(cov_probability, axis=0)
    outage = np.subtract(1, coverage_PR)

    results = {
        'cluster_id':cluster.id, 
        'theta_snr_db': theta_snr_db.tolist(),
        'coverage_PR': coverage_PR.tolist(),
        'outage_PR': outage.tolist(),
        'num_areas': len(snr),
        'num_users_per_area': num_users
    }

    if save_artifacts and savefile_name and directory:
        df = pd.DataFrame([results])
        filename = os.path.join(directory, savefile_name)
        if not os.path.exists(filename): 
            df.to_csv(filename, index=False)
        else: 
            df.to_csv(filename, mode='a', index=False, header=False)

        mymap = np.random.rand(7, 3)
        select = np.random.randint(0, 7)

        image_dir = os.path.join(directory, 'coverage_images') 
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        image_id = uuid.uuid4()
        plt.figure(figsize=(8, 5))
        plt.plot(theta_snr_db, coverage_PR, '-s', color=mymap[select])
        plt.plot(theta_snr_db, outage, '-s', color='black')
        plt.title('Coverage/Outage Probability')
        plt.xlabel('SINR Threshold (dB)')
        plt.ylabel('Coverage Probability')
        plt.legend(['CovPR (H=1250, BW=100MHz)', 'OutPR (H=1250, BW=100MHz)'])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{image_dir}/cov_out_probability_{image_id}.png')
        plt.close()

    return results
