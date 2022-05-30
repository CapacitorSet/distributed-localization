# distributed-localization

Code for the "Modeling and control of cyberphysical systems" course assignments at Politecnico di Torino. Implements IST, DIST, O-DIST.

Use params.py to tune the parameters; run with `-h` to see command line flags.

Creates three files with data for analysis:

 - `ist.csv`, with syntax `seed;num_sensors;connection_distance;RSS_std_dev;failure_chance;stubborn;error;num_iterations` (one line per run)
 - `dist.csv`, with syntax `seed;num_sensors;connection_distance;RSS_std_dev;failure_chance;stubborn;error;num_iterations;essential_spectral_radius` (one line per run)
 - `o-dist.csv`, with syntax `seed;num_sensors;connection_distance;RSS_std_dev;failure_chance;stubborn;i;error;cumulative_error` (one line per target change)
