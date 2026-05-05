import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide", page_title="VBS Monte Carlo Robustness")

def pid_controller(error, error_prev, integral, kp, ki, kd, dt):
    p_term = kp * error
    integral += error * dt
    integral = np.clip(integral, -0.5, 0.5)
    i_term = ki * integral
    if dt > 0:
        d_term = kd * (error - error_prev) / dt
    else:
        d_term = 0
    output = p_term + i_term + d_term
    return output, integral

def simulate_buoyancy_system(depth_command, dt=0.002, step_size=0.00005, kp=0.01, ki=0.001, kd=0.1, k_p=62.409, 
                             a_1=6.8, a_2=0.7, c=0.0, sensor_noise_std=0.0017, sensor_update_rate=10.0,
                             use_sensor_model=True, ma_window_size=5, dist_type="None", dist_amp=0.0, 
                             dist_freq=1.0, dist_start=5.0, noise_power=0.0):
    max_volume_change = 0.0001272
    piston_area = 0.00636177
    max_piston_height = max_volume_change / piston_area
    
    n = len(depth_command)
    t = np.arange(n) * dt
    
    depth = np.zeros(n)
    velocity = np.zeros(n)
    
    error_integral = 0
    error_prev = 0
    
    target_piston_pos = 0.0
    actual_piston_pos = 0.0
    piston_speed = step_size / 0.086
    
    last_measured_depth = 0.0
    ma_buffer = []
    
    if use_sensor_model and sensor_update_rate > 0:
        ticks_per_update = max(1, int(1.0 / (sensor_update_rate * dt)))
    else:
        ticks_per_update = 1
    
    pid_out = 0.0
    
    for i in range(n):
        t_current = t[i]
        if use_sensor_model:
            if i % ticks_per_update == 0:
                std_noise = np.random.normal(0, sensor_noise_std)
                sensor_dt = 1.0 / sensor_update_rate if sensor_update_rate > 0 else dt
                bl_noise = np.random.normal(0, np.sqrt(noise_power / 1000*sensor_dt)) if noise_power > 0 else 0.0
                noise = std_noise + bl_noise
                raw_depth = depth[i] + noise
                ma_buffer.append(raw_depth)
                if len(ma_buffer) > ma_window_size:
                    ma_buffer.pop(0)
                
                last_measured_depth = sum(ma_buffer) / len(ma_buffer)
                
                error = depth_command[i] - last_measured_depth
                pid_out, error_integral = pid_controller(
                    error, error_prev, error_integral, kp, ki, kd, sensor_update_rate > 0 and 1.0/sensor_update_rate or dt
                )
                error_prev = error
        else:
            error = depth_command[i] - depth[i]
            pid_out, error_integral = pid_controller(
                error, error_prev, error_integral, kp, ki, kd, dt
            )
            error_prev = error
            
        desired_piston_pos = np.clip(pid_out, -max_piston_height, max_piston_height)
        
        pos_error = desired_piston_pos - target_piston_pos
        pulse = 0

        # binary operation (either expand or contract)
        if pos_error >= step_size:
            pulse = step_size
        elif pos_error < -step_size:
            pulse = -step_size
            
        target_piston_pos += pulse
        
        pos_diff = target_piston_pos - actual_piston_pos
        max_move = piston_speed * dt
        
        if abs(pos_diff) <= max_move:
            actual_piston_pos = target_piston_pos
        else:
            actual_piston_pos += np.sign(pos_diff) * max_move
            
        # Disturbance
        dist_val = 0.0
        if t_current >= dist_start:
            if dist_type == "Step":
                dist_val = dist_amp
            elif dist_type == "Sinusoidal":
                dist_val = dist_amp * np.sin(2 * np.pi * dist_freq * (t_current - dist_start))
                
        acceleration = (k_p * actual_piston_pos - a_2 * velocity[i] - c * depth[i] + dist_val) / a_1
        
        velocity[i + 1 if i + 1 < n else -1] = velocity[i] + acceleration * dt
        
        if i + 1 < n:
            depth[i + 1] = depth[i] + velocity[i] * dt
            
    return t, depth_command, depth

def samplewave(T=10, dt=0.002):
    t = np.arange(0, T + 1e-5, dt)
    depth_command = np.zeros_like(t)
    depth_command[t >= 2.0] = 1.0
    return t, depth_command

def calculate_metrics(t, depth, command):
    ss_val = 1.0
    overshoot_val = np.max(depth)
    overshoot_pct = max(0, (overshoot_val - ss_val) / ss_val * 100.0)
    
    outside_band = np.where((depth < 0.98) | (depth > 1.02))[0]
    if len(outside_band) > 0 and outside_band[-1] < len(t) - 1:
        settling_time = t[outside_band[-1]] - 2.0
    else:
        settling_time = float('inf')
        
    return overshoot_pct, settling_time

st.title("VBS Monte Carlo Robustness Testing")

col1, col2, col3, col4 = st.columns(4)
with col1:
    kp = st.number_input("Kp", value=0.01, step=0.01, format="%.6f")
with col2:
    ki = st.number_input("Ki", value=0.001, step=0.001, format="%.6f")
with col3:
    kd = st.number_input("Kd", value=0.1, step=0.1, format="%.6f")
with col4:
    sim_time = st.number_input("Simulation Time (s)", value=100, step=1)

st.subheader("Nominal Plant Dynamics")
col_p1, col_p2, col_p3, col_p4 = st.columns(4)
with col_p1:
    k_p_nom = st.number_input("Nominal Plant Gain (K_p)", value=62.409, step=1.0, format="%.3f")
with col_p2:
    a_1_nom = st.number_input("Nominal Mass (a_1)", value=6.8, step=0.1, format="%.2f")
with col_p3:
    a_2_nom = st.number_input("Nominal Damping (a_2)", value=0.7, step=0.1, format="%.2f")
with col_p4:
    c_nom = st.number_input("Nominal Constant (c)", value=0.0, step=0.1, format="%.2f")

st.subheader("Robustness Criteria & Monte Carlo Settings")
col_r1, col_r2, col_r3, col_r4 = st.columns(4)
with col_r1:
    max_overshoot = st.number_input("Max Overshoot (%)", value=10.0, step=1.0)
with col_r2:
    max_settling = st.number_input("Max Settling Time (s)", value=60.0, step=1.0)
with col_r3:
    num_runs = st.number_input("Monte Carlo Runs", value=50, step=10, min_value=1)
with col_r4:
    percentage_val = st.number_input ("% variation", value=10, step=1, min_value=5)

st.subheader("Disturbance Injection")
col_d1, col_d2, col_d3, col_d4 = st.columns(4)
with col_d1:
    d_type = st.selectbox("Type", ["None", "Step", "Sinusoidal"])
with col_d2:
    d_amp = st.number_input("Amplitude", value=0.0, step=0.1)
with col_d3:
    d_freq = st.number_input("Frequency (Hz)", value=1.0, step=0.1)
with col_d4:
    d_start = st.number_input("Start Time (s)", value=5.0, step=1.0)

st.subheader("Sensor Noise Settings")
col_s1, col_s2 = st.columns(2)
with col_s1:
    noise_pow = st.number_input("BL Noise Power", value=0.0, step=0.0001, format="%.4f")
with col_s2:
    base_noise_std = st.number_input("Base Noise Std Dev (mm)", value=1.7, step=0.1)

if st.button("Run Robustness Test", type="primary"):
    dt = 0.002
    step_size = 0.00005
    t, depth_command = samplewave(T=sim_time, dt=dt)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_depths = []
    all_overshoots = []
    all_settling_times = []
    pass_count = 0
    fail_count = 0

    variation_percentage = percentage_val / 100.0
    
    for run in range(int(num_runs)):
        # Randomize parameters +/- variation_percentage %
        k_p_rand = np.random.uniform(k_p_nom * (1 - variation_percentage), k_p_nom * (1 + variation_percentage))
        a_1_rand = np.random.uniform(a_1_nom * (1 - variation_percentage), a_1_nom * (1 + variation_percentage))
        a_2_rand = np.random.uniform(a_2_nom * (1 - variation_percentage), a_2_nom * (1 + variation_percentage))
        c_rand = c_nom # Assuming constant isn't varied
        
        noise_rand = np.random.uniform(base_noise_std * 0.8, base_noise_std * 1.2) / 1000.0
        
        _, _, depth_actual = simulate_buoyancy_system(
            depth_command, dt=dt, step_size=step_size, kp=kp, ki=ki, kd=kd,
            k_p=k_p_rand, a_1=a_1_rand, a_2=a_2_rand, c=c_rand,
            sensor_noise_std=noise_rand, sensor_update_rate=10.0, use_sensor_model=True, ma_window_size=5,
            dist_type=d_type, dist_amp=d_amp, dist_freq=d_freq, dist_start=d_start, noise_power=noise_pow
        )
        
        ov, st_time = calculate_metrics(t, depth_actual, depth_command)
        
        if ov <= max_overshoot and st_time <= max_settling:
            pass_count += 1
        else:
            fail_count += 1
            
        all_depths.append(depth_actual)
        all_overshoots.append(ov)
        all_settling_times.append(st_time if st_time != float('inf') else sim_time)
        
        progress_bar.progress((run + 1) / int(num_runs))
        status_text.text(f"Running simulation {run+1}/{int(num_runs)}...")
        
    status_text.text("Simulation Complete!")
    
    all_depths = np.array(all_depths)
    min_depth = np.min(all_depths, axis=0)
    max_depth = np.max(all_depths, axis=0)
    mean_depth = np.mean(all_depths, axis=0)
    
    st.markdown(f"### Results: {pass_count}/{int(num_runs)} Passed")
    if fail_count == 0:
        st.success("Controller is Highly Robust! All runs passed the criteria.")
    elif pass_count > int(num_runs) * 0.8:
        st.warning(f"Controller is marginally robust. {fail_count} runs failed the criteria.")
    else:
        st.error(f"Controller failed robustness checks. {fail_count} runs failed.")
        
    fig = go.Figure()
    
    # Plot Envelope
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([max_depth, min_depth[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 100, 255, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Monte Carlo Envelope (Min/Max)'
    ))
    
    # Plot Mean
    fig.add_trace(go.Scatter(x=t, y=mean_depth, line=dict(color='blue'), name='Mean Depth'))
    
    # Plot Command
    fig.add_trace(go.Scatter(x=t, y=depth_command, line=dict(color='white', dash='dash'), name='Depth Command'))
    
    # Plot +/- 2% Steady State Bounds
    fig.add_trace(go.Scatter(x=t, y=np.where(t>=2.0, 1.02, 0), line=dict(color='red', width=1, dash='dot'), name='+2% Bound'))
    fig.add_trace(go.Scatter(x=t, y=np.where(t>=2.0, 0.98, 0), line=dict(color='red', width=1, dash='dot'), name='-2% Bound'))
    
    fig.update_layout(title="Monte Carlo Robustness Simulation", xaxis_title="Time (s)", yaxis_title="Depth (m)", height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Overshoot vs. Settling Time Distribution")
    fig2 = go.Figure()
    
    # Colors for pass/fail points
    colors = ['green' if (o <= max_overshoot and s <= max_settling) else 'red' for o, s in zip(all_overshoots, all_settling_times)]
    
    fig2.add_trace(go.Scatter(
        x=all_settling_times, 
        y=all_overshoots,
        mode='markers',
        marker=dict(size=10, color=colors, opacity=0.7, line=dict(width=1, color='white')),
        name="Simulation Runs"
    ))
    
    fig2.add_hline(y=max_overshoot, line_dash="dash", line_color="red", annotation_text="Max Overshoot")
    fig2.add_vline(x=max_settling, line_dash="dash", line_color="red", annotation_text="Max Settling Time")
    
    # Highlight Highest Overshoot
    highest_ov_idx = np.argmax(all_overshoots)
    fig2.add_annotation(
        x=all_settling_times[highest_ov_idx], y=all_overshoots[highest_ov_idx],
        text=f"Highest Overshoot: {all_overshoots[highest_ov_idx]:.1f}%",
        showarrow=True, arrowhead=1, ax=50, ay=-30
    )
    
    # Highlight Highest Settling Time
    highest_st_idx = np.argmax(all_settling_times)
    fig2.add_annotation(
        x=all_settling_times[highest_st_idx], y=all_overshoots[highest_st_idx],
        text=f"Highest Settling Time: {all_settling_times[highest_st_idx]:.1f}s",
        showarrow=True, arrowhead=1, ax=-50, ay=-30
    )
    
    fig2.update_layout(title="Monte Carlo Distribution (Green = Pass, Red = Fail)", xaxis_title="Settling Time (s)", yaxis_title="Overshoot (%)", height=500)
    st.plotly_chart(fig2, use_container_width=True)
