import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib

matplotlib.rcParams['mathtext.fontset']='cm'

def pid_controller(error, error_prev, integral, kp, ki, kd, dt):
    """
    PID controller implementation.
    """
    # Proportional term
    p_term = kp * error
    
    # Integral term with anti-windup
    integral += error * dt
    integral = np.clip(integral, -0.5, 0.5)  # Anti-windup
    i_term = ki * integral
    
    # Derivative term
    if dt > 0:
        d_term = kd * (error - error_prev) / dt
    else:
        d_term = 0
    
    output = p_term + i_term + d_term
    
    return output, integral

def simulate_buoyancy_system(depth_command, dt=0.002, step_size=0.00005):
    """
    Simulate underwater vehicle buoyancy system with closed-loop quantization.
    
    Plant Dynamics:
    G(s) = 62.409 / (6.8s^2 + 0.7s)
    - Input (u): Actual Piston Height [m] (quantized by stepper motor)
    - Output (y): Depth [m]
    """
    # System parameters
    mass = 6.8
    damping = 0.7
    gain = 62.409
    max_volume_change = 0.0001272  # m³
    piston_area = 0.00636177  # Derived from gain / (1000 * 9.81)
    max_piston_height = max_volume_change / piston_area  # approx 0.02 m
    
    # PID gains 
    kp = 0.1
    ki = 0.01
    kd = 0.5
    
    n = len(depth_command)
    t = np.arange(n) * dt
    
    # State variables
    depth = np.zeros(n)
    velocity = np.zeros(n)
    pid_output = np.zeros(n)
    actual_piston_height = np.zeros(n)
    motor_pulses = np.zeros(n)
    
    error_integral = 0
    error_prev = 0
    
    # Initial piston position
    current_piston_pos = 0.0
    
    # Simulation loop
    for i in range(n):
        # Error = desired depth - actual depth
        error = depth_command[i] - depth[i]
        
        # 1. PID controller calculates desired continuous piston position
        pid_out, error_integral = pid_controller(
            error, error_prev, error_integral, kp, ki, kd, dt
        )
        pid_output[i] = pid_out
        error_prev = error
        
        # Saturate desired PID output to max limits
        desired_piston_pos = np.clip(pid_out, -max_piston_height, max_piston_height)
        
        # 2. Delta Modulator (Stepper Motor Logic)
        # Compare current position to desired, send pulse if difference >= step_size
        pos_error = desired_piston_pos - current_piston_pos
        pulse = 0
        if pos_error >= step_size:
            pulse = step_size
        elif pos_error <= -step_size:
            pulse = -step_size
            
        motor_pulses[i] = pulse
        
        # 3. Update actual physical piston position based on pulse
        current_piston_pos += pulse
        actual_piston_height[i] = current_piston_pos
        
        # 4. Plant dynamics: 6.8 * depth_accel + 0.7 * depth_vel = 62.409 * actual_piston_height
        acceleration = (gain * actual_piston_height[i] - damping * velocity[i]) / mass
        
        # Euler integration
        velocity[i + 1 if i + 1 < n else -1] = velocity[i] + acceleration * dt
        
        if i + 1 < n:
            depth[i + 1] = depth[i] + velocity[i] * dt
            
    return t, depth_command, depth, pid_output, actual_piston_height, motor_pulses

def samplewave(T=5, dt=0.002):
    """Generate 1 meter step depth command."""
    t = np.arange(0, T + 1e-5, dt)
    depth_command = np.zeros_like(t)
    depth_command[t >= 2.0] = 1.0
    return t, depth_command

def visualize_pid_dsm_system(t, depth_command, depth_actual, pid_output, actual_piston_height, motor_pulses):
    """
    Visualize PID control with true closed-loop quantization (Delta Modulation).
    """
    fig = plt.figure(figsize=(14, 12), facecolor='white')
    gs = gridspec.GridSpec(5, 1, height_ratios=[1.5, 1, 1, 1, 1], hspace=0.4)
    
    # Plot 1: Depth Response
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, depth_command, 'b--', label='Depth Command', linewidth=2)
    ax1.plot(t, depth_actual, 'g-', label='Actual Depth (m)', linewidth=2)
    ax1.fill_between(t, depth_command, depth_actual, alpha=0.2, color='orange', label='Tracking Error')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_ylabel('Depth (m)', fontsize=11)
    ax1.set_title('Depth Control Response (1m Step Command at t=2s)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels([])
    ax1.set_xlim([min(t), max(t)])
    
    # Plot 2: PID Controller Output
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t, pid_output * 1000, 'r-', linewidth=1.5, label='PID Output (mm)')
    ax2.axhline(y=20, color='k', linestyle='--', alpha=0.5, label='Max Variation (±20 mm)')
    ax2.axhline(y=-20, color='k', linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_ylabel('PID Output (mm)', fontsize=11)
    ax2.set_title('Continuous PID Output (Desired Piston Position)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticklabels([])
    ax2.set_xlim([min(t), max(t)])
    
    # Plot 3: Actual Piston Position
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(t, actual_piston_height * 1000, 'b-', linewidth=1.5, label='Actual Piston Position (Quantized)')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.set_ylabel('Actual Pos (mm)', fontsize=11)
    ax3.set_title('True Piston Position Inside Plant', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticklabels([])
    ax3.set_xlim([min(t), max(t)])
    
    # Plot 4: Delta Modulation Pulse Train
    ax4 = fig.add_subplot(gs[3])
    ax4.step(t, motor_pulses * 1000, 'r-', linewidth=1, label='Motor Step Pulses (±0.05 mm)', where='post')
    ax4.legend(loc='upper right', fontsize=10)
    ax4.set_ylabel('Pulses (mm)', fontsize=11)
    ax4.set_title('Motor Step Commands', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_yticks([-0.05, 0, 0.05])
    ax4.set_ylim([-0.06, 0.06])
    ax4.set_xticklabels([])
    ax4.set_xlim([min(t), max(t)])
    
    # Plot 5: Tracking Error
    ax5 = fig.add_subplot(gs[4])
    tracking_error = depth_command - depth_actual
    ax5.plot(t, tracking_error, 'purple', linewidth=1.5, label='Depth Error (Command - Actual)')
    ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax5.fill_between(t, 0, tracking_error, where=(tracking_error >= 0), alpha=0.3, color='red', label='Undershoot')
    ax5.fill_between(t, 0, tracking_error, where=(tracking_error < 0), alpha=0.3, color='green', label='Overshoot')
    ax5.legend(loc='upper right', fontsize=10)
    ax5.set_ylabel('Error (m)', fontsize=11)
    ax5.set_xlabel('Time (s)', fontsize=11)
    ax5.set_title('Depth Tracking Error', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([min(t), max(t)])
    
    fig.suptitle('PID Control with Closed-Loop Quantization - Underwater Vehicle Depth Control\nPlant: 62.409 / (6.8s² + 0.7s) | Max Vol: 0.0001272 m³ | Step Size: 0.05 mm',
                 fontsize=13, fontweight='bold', y=0.995)
    
    return fig

# ============ Main Execution ============
if __name__ == "__main__":
    dt = 0.002  # Sampling period
    step_size = 0.00005  # m (0.05 mm)
    t, depth_command = samplewave(T=10, dt=dt)  
    
    # Simulate the buoyancy system
    t_sim, cmd, depth_actual, pid_out, actual_piston_height, motor_pulses = simulate_buoyancy_system(depth_command, dt=dt, step_size=step_size)
    
    # Create visualization
    fig = visualize_pid_dsm_system(t_sim, cmd, depth_actual, pid_out, actual_piston_height, motor_pulses)
    
    # Save the figure
    output_path = r'C:\Users\Lucas Barcaro\Dummy\delta_sigma_modulation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    print(f"\n=== Variable Buoyancy System Parameters ===")
    print(f"Plant: G(s) = 62.409 / (6.8s^2 + 0.7s)")
    print(f"Step Size: {step_size * 1000} mm")
    plt.show()