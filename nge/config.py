"""Game constants shared by the classic simulation, the neural model, server and client.

World coordinates are normalised to the unit square: x to the right, y downward
(so invaders live near y=0 and ships near y=1, like Space Invaders).
"""

TICK_RATE = 30
DT = 1.0 / TICK_RATE

N_PLAYERS = 2
N_BULLETS = 3            # bullet slots per ship
INV_COLS = 6
INV_ROWS = 3
N_INVADERS = INV_COLS * INV_ROWS

# ---- ships -------------------------------------------------------------
SHIP_RADIUS = 0.03
SHIP_ACCEL = 2.5          # units / s^2 at full stick deflection
SHIP_DAMPING = 0.90       # per-tick velocity multiplier
SHIP_MAX_SPEED = 0.8      # units / s
SHIP_X_MIN = SHIP_RADIUS
SHIP_X_MAX = 1.0 - SHIP_RADIUS
SHIP_Y_MIN = 0.62         # ships are confined to the bottom band
SHIP_Y_MAX = 1.0 - SHIP_RADIUS

# ---- bullets -----------------------------------------------------------
BULLET_SPEED = 1.2        # units / s, upward
FIRE_COOLDOWN_DECAY = 0.1 # cooldown goes 1 -> 0 in 10 ticks

# ---- invader formation -------------------------------------------------
INV_X0 = 0.08
INV_Y0 = 0.08
INV_DX = 0.12
INV_DY = 0.08
INV_HALF_W = 0.035
INV_HALF_H = 0.03
FORM_SPEED = 0.15         # units / s horizontal
FORM_OX_MAX = 0.28
FORM_STEP_DOWN = 0.04
FORM_OY_MAX = 0.24
