import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide", page_title="VBS Simulation")

def pid_controller(error, error_prev, integral, kp, ki, kd, dt):
    p_term = kp * error
    integral += error * dt
    # saturation
    integral = np.clip(integral, -0.5, 0.5)
    i_term = ki * integral
    if dt > 0:
        d_term = kd * (error - error_prev) / dt
    else:
        d_term = 0
    output = p_term + i_term + d_term
    return output, integral

def simulate_buoyancy_system(depth_command, dt=0.002, step_size=0.00005, kp=0.001, ki=0.0001, kd=0.5, k_p=62.409, a_1=6.8, a_2=0.7, c=0.0, sensor_noise_std=0.0017, sensor_update_rate=10.0, use_sensor_model=True, ma_window_size=1):
    max_volume_change = 0.0001272
    piston_area = 0.00636177
    max_piston_height = max_volume_change / piston_area
    
    n = len(depth_command)
    t = np.arange(n) * dt
    
    depth = np.zeros(n)
    velocity = np.zeros(n)
    pid_output = np.zeros(n)
    actual_piston_height = np.zeros(n)
    motor_pulses = np.zeros(n)
    measured_depth = np.zeros(n)
    
    error_integral = 0
    error_prev = 0
    
    target_piston_pos = 0.0
    actual_piston_pos = 0.0
    piston_speed = step_size / 0.086  # Speed to move 0.05mm in 86ms
    
    last_measured_depth = 0.0
    ma_buffer = []
    
    if use_sensor_model and sensor_update_rate > 0:
        ticks_per_update = max(1, int(1.0 / (sensor_update_rate * dt)))
    else:
        ticks_per_update = 1
    
    pid_out = 0.0
    
    for i in range(n):
        if use_sensor_model:
            if i % ticks_per_update == 0:
                noise = np.random.normal(0, sensor_noise_std)
                raw_depth = depth[i] + noise
                
                # Apply Moving Average Filter
                ma_buffer.append(raw_depth)
                if len(ma_buffer) > ma_window_size:
                    ma_buffer.pop(0)
                
                last_measured_depth = sum(ma_buffer) / len(ma_buffer)
                
                # Run PID ONLY when a new sensor reading arrives
                error = depth_command[i] - last_measured_depth
                pid_out, error_integral = pid_controller(
                    error, error_prev, error_integral, kp, ki, kd, sensor_update_rate > 0 and 1.0/sensor_update_rate or dt
                )
                error_prev = error
                
            measured_depth[i] = last_measured_depth
        else:
            measured_depth[i] = depth[i]
            error = depth_command[i] - depth[i]
            pid_out, error_integral = pid_controller(
                error, error_prev, error_integral, kp, ki, kd, dt
            )
            error_prev = error
            
        pid_output[i] = pid_out
        
        # saturation
        desired_piston_pos = np.clip(pid_out, -max_piston_height, max_piston_height)
        
        pos_error = desired_piston_pos - target_piston_pos
        pulse = 0

        # binary operation (either expand or contract)
        if pos_error >= step_size:
            pulse = step_size
        elif pos_error < -step_size:
            pulse = -step_size
            
        motor_pulses[i] = pulse
        target_piston_pos += pulse
        
        # Transient piston movement (constant speed)
        pos_diff = target_piston_pos - actual_piston_pos
        max_move = piston_speed * dt
        
        # saturation
        if abs(pos_diff) <= max_move:
            actual_piston_pos = target_piston_pos
        else:
            actual_piston_pos += np.sign(pos_diff) * max_move
            
        actual_piston_height[i] = actual_piston_pos
        
        acceleration = (k_p * actual_piston_height[i] - a_2 * velocity[i] - c * depth[i]) / a_1
        
        velocity[i + 1 if i + 1 < n else -1] = velocity[i] + acceleration * dt
        
        if i + 1 < n:
            depth[i + 1] = depth[i] + velocity[i] * dt
            
    return t, depth_command, depth, pid_output, actual_piston_height, motor_pulses, measured_depth

def samplewave(T=10, dt=0.002):
    t = np.arange(0, T + 1e-5, dt)
    depth_command = np.zeros_like(t)
    depth_command[t >= 2.0] = 1.0
    return t, depth_command

st.title("VBS Closed-Loop Delta Modulation Simulation")

col1, col2, col3, col4 = st.columns(4)
with col1:
    kp = st.number_input("Kp", value=0.001, step=0.01, format="%.6f")
with col2:
    ki = st.number_input("Ki", value=0.0001, step=0.001, format="%.6f")
with col3:
    kd = st.number_input("Kd", value=0.5, step=0.1, format="%.6f")
with col4:
    sim_time = st.number_input("Simulation Time (s)", value=100, step=1)

st.subheader("Plant Dynamics")
col_p1, col_p2, col_p3, col_p4 = st.columns(4)
with col_p1:
    k_p_input = st.number_input("Plant Gain (K_p)", value=62.409, step=1.0, format="%.3f")
with col_p2:
    a_1_input = st.number_input("a_1 (s^2 term)", value=6.8, step=0.1, format="%.2f")
with col_p3:
    a_2_input = st.number_input("a_2 (s term)", value=0.7, step=0.1, format="%.2f")
with col_p4:
    c_input = st.number_input("c (constant term)", value=0.0, step=0.1, format="%.2f")

st.subheader("Sensor Model (Real World Characterization)")
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    use_sensor = st.checkbox("Enable Sensor Model", value=True)
with col_s2:
    noise_std = st.number_input("Noise Std Dev (mm)", value=1.7, step=0.1)
with col_s3:
    sensor_rate = st.number_input("Sensor Update Rate (Hz)", value=10.0, step=1.0)
with col_s4:
    ma_window = st.number_input("Moving Average Window", value=5, step=1, min_value=1)

dt = 0.002
step_size = 0.00005

t, depth_command = samplewave(T=sim_time, dt=dt)
t_sim, cmd, depth_actual, pid_out, actual_piston_height, motor_pulses, measured_depth = simulate_buoyancy_system(
    depth_command, dt=dt, step_size=step_size, kp=kp, ki=ki, kd=kd,
    k_p=k_p_input, a_1=a_1_input, a_2=a_2_input, c=c_input,
    sensor_noise_std=noise_std / 1000.0, sensor_update_rate=sensor_rate, use_sensor_model=use_sensor, ma_window_size=ma_window
)

fig = make_subplots(
    rows=5, cols=1, 
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=(
        "Depth Response (m)", 
        "Desired Continuous PID Output (mm)", 
        "Actual Quantized Piston Position (mm)",
        "Motor Step Pulses (mm)",
        "Depth Tracking Error (m)"
    )
)

# Plot 1: Depth Response
fig.add_trace(go.Scatter(x=t_sim, y=cmd, name="Depth Command", line=dict(color='white', dash='dash')), row=1, col=1)
fig.add_trace(go.Scatter(x=t_sim, y=depth_actual, name="Actual Depth", line=dict(color='green')), row=1, col=1)
if use_sensor:
    fig.add_trace(go.Scatter(x=t_sim, y=measured_depth, name="Measured Depth", line=dict(color='orange', width=1, dash='dot'), opacity=0.7), row=1, col=1)

# Plot 2: PID Output
fig.add_trace(go.Scatter(x=t_sim, y=pid_out*1000, name="PID Output (mm)", line=dict(color='red')), row=2, col=1)

# Plot 3: Actual Piston Position
fig.add_trace(go.Scatter(x=t_sim, y=actual_piston_height*1000, name="Actual Piston Pos (mm)", line=dict(color='white')), row=3, col=1)

# Plot 4: Motor Pulses
fig.add_trace(go.Scatter(x=t_sim, y=motor_pulses*1000, name="Pulses (mm)", line=dict(color='red', shape='hv')), row=4, col=1)

# Plot 5: Tracking Error
fig.add_trace(go.Scatter(x=t_sim, y=cmd - depth_actual, name="Error (m)", line=dict(color='purple')), row=5, col=1)

fig.update_layout(height=1000, title_text="VBS Simulation Interactive Dashboard", showlegend=True)
st.plotly_chart(fig, use_container_width=True)
