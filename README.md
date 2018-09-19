## Perform Iterative LQR with obstacle avoidance using linearized ellipsoid propagation
Main algorithm in `gn_lqr_algo.py`.

## Installation
Place the folder in your python path. Replace `scipy/optimize/_trf.py` with
`updated_trf.py` provided that includes options for maxiters and callback
function that gets called after every iteration for trf


## Examples
Example usage in `gn_lqr_linear_dynamics.py`, `gn_lqr_unicycle_dynamics.py`.
Tests in nose2
