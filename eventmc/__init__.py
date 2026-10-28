"""EventMC: training-free motion-corrected 3D brain MRI reconstruction via event-guided references."""
from .core import Core
from .anchor import prerun, guard
from .stages import stage_a, stage_b, final_reconstruction
from .pipeline import run_eventmc, eventmc_s, eventmc_half, half_resolution

__all__ = ['Core', 'prerun', 'guard', 'stage_a', 'stage_b', 'final_reconstruction', 'run_eventmc', 'eventmc_s',
           'eventmc_half', 'half_resolution']
__version__ = '1.0.0'
