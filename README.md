# Admittance Control Benchmark

## Installation and Dependencies

The source code was developed on Ubuntu 22.04 with:

1. Python 3.10
2. [Pinocchio](https://stack-of-tasks.github.io/pinocchio/download.html#Install_1) 2.6.20
3. Matplotlib 3.6.2
4. Numpy 1.23.5
5. example-robot-data 4.3.0
6. meshcat

## Usage

The simulation setup was design to rely on the `MeshcatVisualizer` running
on a web browser. Once the simulation starts a new window on your browser will
display the robot arm and peform the movement according to the [simulation script](./controllers/impedance_6dof.py).

Browser window url: `http://127.0.0.1:7000/static/`

Run the simulation on interactive mode for better experience:

```sh
python3 -i controllers/clear_board_clean.py
```

Simulation logs are stored in the [data](./data/) directory. Graphs using these logs can be seen on [plots](./plots/).
Please consider this directory struct for your usage and development. Data files are here for exemplification only.
**Do not open PR with data/log files.**
