"""EventMC: training-free motion-corrected 3D brain MRI reconstruction via event-guided references.

Main entry point: run_eventmc(y, segments, ...). Module overview:
  params    algorithm parameters used in the paper
  model     forward model, ImgEst (EncodingModel.reconstruct) and MotEst (EncodingModel.fit_pose, fit_poses_outward)
  events    event detection and event-aware pose smoothing
  stages    Stage A, Stage B and the final reconstruction (Algorithm 1)
  anchor    pre-run, within-segment check and anchor selection
  pipeline  EventMC-S and the multi-anchor extension (run_eventmc)
"""
from .model import EncodingModel
from .events import detect_events, smooth_poses, rms_displacement
from .anchor import prerun, guard
from .stages import stage_a, stage_b, final_reconstruction, stable_interval
from .pipeline import run_eventmc, eventmc_s, eventmc_half, half_resolution

__all__ = ['run_eventmc', 'EncodingModel', 'prerun', 'guard', 'stage_a', 'stage_b', 'final_reconstruction',
           'stable_interval', 'detect_events', 'smooth_poses', 'rms_displacement', 'eventmc_s', 'eventmc_half',
           'half_resolution']
__version__ = '1.0.0'
