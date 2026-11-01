"""Algorithm parameters used in the paper (appendix, Table A.2).

All poses are (rotation vector in degrees, translation in mm)."""

# Objective, Eq. (2)
TV_WEIGHT = 1e-3            # TV weight lambda for normalised data (CC359); 3e-3 was used for PMoC3D
TV_EPS = 1e-6               # smoothing constant inside the TV square root

# L-BFGS budgets
IMAGE_ITERS = 15            # ImgEst in Stages A and B
POSE_ITERS = 30             # MotEst, per segment and starting pose
FINAL_ITERS = 30            # final reconstruction (Algorithm 1, line 13)

# Stage A (Section 3.1)
STAGE_A_ROUNDS = 5
GROUPS = 4                  # references omit the target's group s mod 4 (Eq. (3))

# Stage B (Section 3.2): (central fraction f of each k-space axis, number of rounds) -> f = 0.25, 0.5, 1, 1
STAGE_B_SCHEDULE = ((0.25, 1), (0.5, 1), (1.0, 2))

# Event detection (Section 3.1, appendix A.4)
PROBE_RADIUS_MM = 60.0      # six probe points at +-60 mm along each axis
EVENT_MAD_FACTOR = 3.0      # event if m_j > max(3 sigma_j, 0.6 mm)
EVENT_FLOOR_MM = 0.6
EVENT_NEIGHBOURS = 3        # sigma_j from up to three boundaries on each side

# Event-aware smoothing (appendix A.4)
SMOOTH_SIGMA = 0.3          # data term weight 1 / sigma^2
SMOOTH_LAMBDA = 1 / 0.15 ** 2
SMOOTH_DELTA = 0.3          # Huber threshold on second differences
SMOOTH_MAX_ITERS = 30       # IRLS iterations

# Within-segment check ("Guard", pre-run on CC359; appendix A.2)
GUARD_SLICES = 24
GUARD_WINDOWS = 3           # sub-windows per segment
GUARD_Z = 3.0
GUARD_MAX_ROUNDS = 5
GUARD_ITERS = 40            # L-BFGS iterations of each 2D TV prediction

# Extra translation starts for segments whose phase-encoding period is shorter than this (mm). Interleaved segments
# span all of k-space, so this never applies to the acquisitions EventMC is designed for; it is kept to reproduce the
# paper's code.
ALIAS_MAX_PERIOD = 5.0
