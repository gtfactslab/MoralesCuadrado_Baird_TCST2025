import jax.numpy as jnp
from jax import jit, jacfwd, lax, jacrev, hessian

GRAVITY = 9.806
USING_CBFS = True
C = jnp.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 1]])


# ~~~ Functions for Tracking Error Calculation Using Quaternions for Yaw ~~~
# @jit
def quaternion_from_yaw(yaw):
    """Converts a yaw angle to a quaternion."""
    half_yaw = yaw / 2.0
    return jnp.array([jnp.cos(half_yaw), 0, 0, jnp.sin(half_yaw)])

# @jit
def quaternion_conjugate(q):
    """Returns the conjugate of a quaternion."""
    return jnp.array([q[0], -q[1], -q[2], -q[3]])

# @jit
def quaternion_multiply(q1, q2):
    """Multiplies two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

# @jit
def yaw_error_from_quaternion(q):
    """Returns the yaw error from the quaternion of angular error."""
    return 2 * jnp.arctan2(q[3], q[0])

# @jit
def quaternion_normalize(q):
    """Normalizes a quaternion."""
    return q / jnp.linalg.norm(q)

# @jit
def shortest_path_yaw_quaternion(current_yaw, desired_yaw):
    """Returns the shortest path between two yaw angles with quaternions."""
    q_current = quaternion_normalize(quaternion_from_yaw(current_yaw))
    q_desired = quaternion_normalize(quaternion_from_yaw(desired_yaw))
    q_error = quaternion_multiply(q_desired, quaternion_conjugate(q_current))
    q_error_normalized = quaternion_normalize(q_error)
    return yaw_error_from_quaternion(q_error_normalized)

# @jit
def get_tracking_error(ref, pred):
    """Calculates the tracking error between the reference and predicted outputs with yaw error handled by quaternions."""
    err = ref - pred
    err = err.at[3, 0].set(shortest_path_yaw_quaternion(pred[3, 0], ref[3, 0]))
    return err



# ~~~ Functions for Newton-Raphson Flow Using Nonlinear FWD Euler Prediction ~~~
# @jit
def dynamics(state, input, mass):
    """Quadrotor dynamics. xdot = f(x, u)."""
    x, y, z, vx, vy, vz, roll, pitch, yaw = state
    curr_thrust = input[0]
    body_rates = input[1:]
    T = jnp.array([[jnp.array([1]), jnp.sin(roll) * jnp.tan(pitch), jnp.cos(roll) * jnp.tan(pitch)],
                    [jnp.array([0]), jnp.cos(roll), -jnp.sin(roll)],
                    [jnp.array([0]), jnp.sin(roll) / jnp.cos(pitch), jnp.cos(roll) / jnp.cos(pitch)]]).squeeze()
    curr_rolldot, curr_pitchdot, curr_yawdot = T @ body_rates

    sr = jnp.sin(roll)
    sy = jnp.sin(yaw)
    sp = jnp.sin(pitch)
    cr = jnp.cos(roll)
    cp = jnp.cos(pitch)
    cy = jnp.cos(yaw)

    vxdot = -(curr_thrust / mass) * (sr * sy + cr * cy * sp)
    vydot = -(curr_thrust / mass) * (cr * sy * sp - cy * sr)
    vzdot = GRAVITY - (curr_thrust / mass) * (cr * cp)

    xdot = jnp.array([vx, vy, vz, vxdot, vydot, vzdot, curr_rolldot, curr_pitchdot, curr_yawdot]).reshape((9,1))
    return xdot

# @jit
def fwd_euler(state, input, lookahead_step, integrations_int, mass):
    """Forward Euler integration."""
    def for_function(i, current_state):
        return current_state + dynamics(current_state, input, mass) * lookahead_step

    pred_state = lax.fori_loop(0, integrations_int, for_function, state)
    return pred_state

# @jit
def predict_state(state, u, T_lookahead, lookahead_step, mass):
    """Predict the next state at time t+T via fwd euler integration of nonlinear dynamics."""
    integrations_int = (T_lookahead / lookahead_step).astype(int)
    # integrations_int = int(T_lookahead / lookahead_step)
    pred_state = fwd_euler(state, u, lookahead_step, integrations_int, mass)
    return pred_state

# @jit
def predict_output(state, u, T_lookahead, lookahead_step, mass):
    """Take output from the predicted states."""
    pred_state = predict_state(state, u, T_lookahead, lookahead_step, mass)
    return C @ pred_state

# @jit
def get_jac_pred_u(state, last_input, T_lookahead, lookahead_step, mass):
    """Get the jacobian of the predicted output with respect to the control input."""
    raw_val = jacfwd(predict_output, 1)(state, last_input, T_lookahead, lookahead_step, mass)
    return raw_val.reshape((4,4))

# @jit
def get_inv_jac_pred_u(state, last_input, T_lookahead, lookahead_step, mass):
    """Get the inverse of the jacobian of the predicted output with respect to the control input."""
    return jnp.linalg.pinv(get_jac_pred_u(state, last_input, T_lookahead, lookahead_step, mass).reshape((4,4)))

# @jit
def execute_cbf(current, phi, max_value, min_value, gamma):
    """Execute the control barrier function."""
    zeta_max = gamma * (max_value - current) - phi
    zeta_min = gamma * (min_value - current) - phi
    v = jnp.where(current >= 0, jnp.minimum(0, zeta_max), jnp.maximum(0, zeta_min))
    return v

# @jit
def integral_cbf(last_input, phi):
    """Integral control barrier function set-up for all inputs."""
    # Extract values from input
    curr_thrust, curr_roll_rate, curr_pitch_rate, curr_yaw_rate = last_input[:, 0]
    phi_thrust, phi_roll_rate, phi_pitch_rate, phi_yaw_rate = phi[:, 0]

    # CBF parameters
    thrust_gamma = 1.0  # CBF parameter
    thrust_max = 27.0  # max thrust (force) value
    thrust_min = 0.5  # min thrust (force) value
    v_thrust = execute_cbf(curr_thrust, phi_thrust, thrust_max, thrust_min, thrust_gamma)

    # CBF for rates
    rates_max_abs = 0.8  # max absolute value of roll, pitch, and yaw rates
    rates_max = rates_max_abs
    rates_min = -rates_max_abs
    gamma_rates = 1.0  # CBF parameter
    
    v_roll = execute_cbf(curr_roll_rate, phi_roll_rate, rates_max, rates_min, gamma_rates)
    v_pitch = execute_cbf(curr_pitch_rate, phi_pitch_rate, rates_max, rates_min, gamma_rates)
    v_yaw = execute_cbf(curr_yaw_rate, phi_yaw_rate, rates_max, rates_min, gamma_rates)

    v = jnp.array([[v_thrust], [v_roll], [v_pitch], [v_yaw]])
    return v

@jit
def NR_tracker_original(currstate, currinput, ref, T_lookahead, lookahead_step, integration_step, mass):
    """Standard Newton-Raphson method to track the reference trajectory with forward euler integration of dynamics for prediction."""
    alpha = jnp.array([20, 30, 30, 30]).reshape((4,1))
    pred = predict_output(currstate, currinput, T_lookahead, lookahead_step, mass)
    error = get_tracking_error(ref, pred) # calculates tracking error
    dgdu = get_jac_pred_u(currstate, currinput, T_lookahead, lookahead_step, mass)
    dgdu_inv = jnp.linalg.inv(dgdu)

    NR = dgdu_inv @ error # calculates newton-raphson control input without speed-up parameter
    v = integral_cbf(currinput, NR)
    udot = NR + v # udot = { inv(dg/du) * (yref - ypred) } + v = NR + v = newton-raphson control input + CBF adjustment
    change_u = udot * integration_step #crude integration of udot to get u (maybe just use 0.02 as period)
    u = currinput + alpha * change_u # u_new = u_old + alpha * change_u
    
    return u, v



# ~~~ Functions for Newton-Raphson Flow Using Linearized Closed-Form Predictor ~~~
@jit
def linear_predictor(STATE, INPUT, MASS, eAT, int_eATB):
    gravity = jnp.array([[MASS * GRAVITY, 0, 0, 0]]).T #gravity vector that counteracts input vector: [mg, 0, 0, 0]
    return C @ (eAT@STATE + int_eATB @ (INPUT - gravity)) # y(t+T) = C * (eAT * x(t) + int_eATB * (u(t) - gravity))

@jit
def NR_tracker_linpred(currstate, currinput, ref, integration_step, mass, eAT, int_eATB, jac_inv):
    """Newton-Raphson method with linearized prediction for tracking"""
    alpha = jnp.array([20, 30, 30, 30]).reshape((4,1))
    pred = linear_predictor(currstate, currinput, mass, eAT, int_eATB)
    error = get_tracking_error(ref, pred) # calculates tracking error

    NR = jac_inv @ error # calculates newton-raphson control input without speed-up parameter
    v = integral_cbf(currinput, NR)
    udot = NR + v # udot = { inv(dg/du) * (yref - ypred) } + v = NR + v = newton-raphson control input + CBF adjustment
    change_u = udot * integration_step #crude integration of udot to get u (maybe just use 0.02 as period)
    u = currinput + alpha * change_u # u_new = u_old + alpha * change_u
    return u, v

