"""Flying the planned paths: Gazebo (gz Harmonic) + PX4 SITL, or a point-mass dry run.

    mission  -> plan on the predicted field, make it flyable, write the Gazebo world
    fly      -> launch PX4 + Gazebo, fly the reference with MAVSDK offboard, log it
    dryrun   -> the same, with a point-mass tracker instead of PX4 (no install needed)
    analyse  -> regret of the planned, reference and FLOWN paths under the true field
    batch    -> all of the above over many blocks, models and instructions
    report   -> tables

See `python -m ig3d.sim.run --help`.
"""
