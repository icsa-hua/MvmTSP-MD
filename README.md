# Energy-Aware MVMTSP Scheduling for UAV Swarms NTN-stations in Challenging Environments 

## An optimization framework to always determine the optimal navigation for all 
 agents in critical environments (Search and Rescue, SAR operations). 

![License](https://img.shields.io/badge/license-GPL-blue.svg)
![Version](https://img.shields.io/badge/0.2.0-brightgreen.svg)

## Table of Contents

* [Introduction](#introduction)
* [Technologies - Requirements](#technologies)
* [SetUp](#setup)
* [Usage](#usage)
* [License](#license)

## Introduction

This repository is a Two-Phase solution that simulates and optimizes the deployement 
of UAV swarms to service areas with ground users that require communication coverage. 
Simulation includes, map and user generation with world-like movements and locations, 
whereas the optimization consists of a combinatorial optimization problem formulation
known as **Many Visits Multi-Agent Travelling Salesman Problem (MVMTSP)**, that 
respects energy, time synchronization and objective constraints to determine a 
coordinated plan for all agents. 

The main contributions of this repository can be summed up as: 
* Two distinct use cases: Collaborative Plan vs Individual Plan
* MILP MVMTSP formulation with custom constraints 
* Single-Stage vs Two-Stage Objective solution 
* Two core objective functions 
* Custom offline energy consumption estimation model 
* Regionalization based on energy limitations
* Virtual nodes generation and installation 
* Custom heuristic to initialize paths and estimate the time frame for each cluster
* Custom coverage estimation model that considers different environment types 
* Map generation based on real geographical latitude and longitude 
* Ground user generation with realistic pedestrian mobility

## Technologies - Requirements
The main technologies used for this project are:

* Python version: >= 3.9 
* PuLP version >= 2.8
* GLPK-utils version >= 5.0-1
* DEAP version >= 1.4.1
* K-means constrained >= 0.7.5
* Geopy >= 2.4.1 
* Pyproj >= 3.7.1
* Simpy >= 4.1.1 

> [!NOTE]
> This framework has been tested on both Mac OS 
> and Windows OS through WSL2.  

> [!IMPORTANT]
> You will need to install GNU GLPK on the system, and the python package 
> which enables it. Then PuLP will be available to call it 


## SetUp

1. Clone the repository from the default main branch:

```sh
git clone https://github.com/icsa-hua/MvmTSP-MD.git
```

2. Navigate to the project directory:

```sh
cd MvmTSP-MD
```

3. Install the repository as a package through the ```setup.py``` file:

```sh
pip install -e .
```
This installs the MvmTSP-MD as a package with all the dependencies required.  

> Make sure you have the GLPK library installed:

* For Windows system follow the guide: https://winglpk.sourceforge.net/. Or Use WSL2.
* For Linux system use: `sudo apt-get install glpk-utils`
* For Mac os you can use Homebrew to install it with: ```brew install glpk```

## Usage

The framework can be executed through either the _bash scripts_ or through the 
execution_script.py. In both cases you can alter the configuration of the simulation 
and the combinatorial parameters as arguments. We give you the option to experiment 
with different agent, user, nodes population, environment type, objective function 
and stage solution. You can also change the coordinates to include your own. 

>[!NOTE]
> To change the coordinates you need to have the in WGS84 or EPSG 32633. 
> These can be easily obtained with google maps as an example. 
> We handle the transformation to the UTM coordinate system to handle realistic 
> movements. If you wanna learn more, take a look into the --designs/voronoi_map.py--

To execute the simulations in an efficient manner we include two bash scripts 
to represent the two main use cases which you should alter to see how the model 
behaves with different configuration parameters.
Use: 
```sh
./experiments.sh
```
or: 
```sh
./experiments_indi.sh
```
Finally use: ```python3 execution_script.py --args ... --args``` to execute a single 
scenario. 

### Code Structure 
The core is located inside dummy_app directory. 
There are three main sub-directories: **designs**, **models**, **tools**
>1. *designs* -> core classes to represent agents, users, map generation and the problem 
>   builder interface. Also has the _constraints.py_ which holds all of the constraints that
>   create the combinatorial problem. 

>2. *models* -> This holds the _simulation_builder.py_ which combines regionalization, cluster 
>   management, agent allocation and proglem solution with post-processing (interpolation, 
>   synchronization and node transformation into coordinates). This sub-directory also 
>   includes the energy and coverage models, and the TOPSIS and GA formulations. 

>3. *tools* -> This sub-directory creates the logger and the common functions that are 
>   used inside multiple files. We also include a class generation to instantiate 
>   performance metrics, to interface it with the problem builder. 


### Assets 
Inside the _assets_ directory once the problem is executed, you will see a couple of sub-directories.
These include results, logs and scalers. We also save the distance, energy and time cost that are 
produced after the voronoi map generation. 

A) ```/animations``` has the .mp4 files showing the simulation with the paths for the **first** problem 
solution 
B) ```/data``` has the cost after map generation 
C) ```/logs``` has the log after an execution 
D) ```/results``` .csv and .png results captured during simulation 
E) ```/scalers``` the scalers used to normalize the data. You can use them to inverse transform the data 
and use them how you want. 


