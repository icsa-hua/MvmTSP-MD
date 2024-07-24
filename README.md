# Many-Visits Multi-Travelling Salesman Problem with Multiple-Depots (MvmTSP-MD) 

![License](https://img.shields.io/badge/license-GPL-blue.svg)

## Table of Contents 
* [Introduction](#introduction)
* [Technologies](#technologies)
* [SetUp](#setup)
* [Usage](#usage)
* [License](#license)

## Introduction

Welcome to the **MvmTSP-MD**! This repository creates a Swarm of UAVs focused 
optimization problem, considering distance and energy as the optimization variables. 
We construct a unique MvmTSP-MD optimization problem, with a set of constraints, 
focusing on the mission scheduling of each swarm. 

We utilize a custom Genetic Algorithm to streamline the performance of the GLPK 
solver, by initializing the decision variable with the best path for each agent. 
To further address the complexity and computational bottlnecks, a regionalization 
approach is followed for the entire search space based on geographical characteristics. 

We provide the option to toggle every feature and change the complexity of the modulation 
with your specific constraints, areas, agents etc. 

## Technologies

The main technologies used for this project are: 
* Python version: 3.10
* PuLP version: 2.8
* GLPK-utils version: 5.0-1
* DEAP version: 1.4.1
 
## SetUp
1. Clone the repository:
```sh
git clone https://github.com/icsa-hua/MvmTSP-MD.git
```
2. Navigate to the project directory:
```sh
cd MvmTSP-MD
```
3. Install the dependencies
```sh
pip install -r requirements.txt
```

Make sure you have the GLPK library installed: 
* For Windows system follow the guide: https://winglpk.sourceforge.net/. Or Use WSL2.
* For Linux system use: `sudo apt-get install glpk-utils`

## Usage 
Launch the application: `python3 main.py`

#### Data Initialization
Every major data is provided through csv files. Consider the following format: 
* Areas -> Columns of DataFrame X_coords, Y_coords, ID
* Distances -> A NxN array with distance between every pair of nodes
* Energy -> A NxN array with distance between every pair of nodes
* Customers -> Area IDs that containt users 

## License 
This project is licensed under the GPL License. See the [LICENSE](LICENSE) file for details.
