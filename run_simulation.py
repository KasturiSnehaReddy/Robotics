import time
import math
import mujoco
import mujoco.viewer
from path_planning import build_warehouse_graph
from cbs import CBS

class LocalAvoidance:
    """A simplified local collision avoidance (emulating RVO) to prevent physical clipping."""
    def __init__(self, neighbor_dist=1.5, repulsion_strength=3.0):
        self.neighbor_dist = neighbor_dist
        self.repulsion = repulsion_strength
        self.agents = {}
        
    def update_agent(self, agent_id, pos, pref_vel):
        self.agents[agent_id] = {'pos': pos, 'pref_vel': pref_vel, 'safe_vel': (0,0)}
        
    def compute_safe_velocities(self, max_speed):
        for idx, agent in self.agents.items():
            vx, vy = agent['pref_vel']
            px, py = agent['pos']
            
            rx, ry = 0.0, 0.0
            for other_idx, other in self.agents.items():
                if idx == other_idx: continue
                
                ox, oy = other['pos']
                dx = px - ox
                dy = py - oy
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist < self.neighbor_dist and dist > 0.01:
                    # Exponential repulsion as they get closer
                    force = self.repulsion / (dist ** 2)
                    rx += (dx / dist) * force
                    ry += (dy / dist) * force
            
            # Combine preferred velocity with avoidance repulsion
            final_vx = vx + rx
            final_vy = vy + ry
            
            # Cap at max speed
            speed = math.sqrt(final_vx**2 + final_vy**2)
            if speed > max_speed:
                final_vx = (final_vx / speed) * max_speed
                final_vy = (final_vy / speed) * max_speed
                
            agent['safe_vel'] = (final_vx, final_vy)

def main():
    print("=== Step 1: Multi-Robot CBS Planning ===")
    G, obstacles = build_warehouse_graph()
    
    starts = {'robot1': (0, -8), 'robot2': (0, 8)}
    goals = {'robot1': (0, 8), 'robot2': (0, -8)}
    
    print("Running Conflict-Based Search...")
    cbs = CBS(G)
    paths = cbs.solve(starts, goals)
    
    if not paths:
        print("ERROR: CBS failed to find paths!")
        return

    print("CBS Paths calculated successfully!")
    
    print("=== Step 2: Body (Mujoco Physics + Local Avoidance) ===")
    model = mujoco.MjModel.from_xml_path("warehouse.xml")
    data = mujoco.MjData(model)
    
    robot_qvel_idx = {'robot1': 0, 'robot2': 6}
    robot_qpos_idx = {'robot1': 0, 'robot2': 7}
    
    MAX_SPEED = 2.0 
    SECONDS_PER_STEP = 1.0 
    
    # Initialize our Local Avoidance system (acting as RVO2)
    avoider = LocalAvoidance(neighbor_dist=1.5, repulsion_strength=1.5)
    
    print("Launching Viewer. Watch the robots dodge each other smoothly!")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            sim_time = data.time
            current_cbs_step = int(sim_time / SECONDS_PER_STEP)
            
            # 1. Calculate Preferred Velocities for all robots based on CBS
            for agent, path in paths.items():
                pos_idx = robot_qpos_idx[agent]
                current_x = data.qpos[pos_idx]
                current_y = data.qpos[pos_idx + 1]
                
                if current_cbs_step < len(path):
                    target_x, target_y = path[current_cbs_step]
                else:
                    target_x, target_y = path[-1]
                
                dist = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
                
                if dist < 0.1:
                    pref_vx, pref_vy = 0.0, 0.0
                else:
                    pref_vx = (target_x - current_x) / dist * MAX_SPEED
                    pref_vy = (target_y - current_y) / dist * MAX_SPEED
                    
                # Feed to our local collision avoidance system
                avoider.update_agent(agent, (current_x, current_y), (pref_vx, pref_vy))
                
            # 2. Compute Safe Velocities considering nearby robots
            avoider.compute_safe_velocities(MAX_SPEED)
            
            # 3. Apply Safe Velocities to Physics Engine
            for agent in paths.keys():
                vel_idx = robot_qvel_idx[agent]
                safe_vx, safe_vy = avoider.agents[agent]['safe_vel']
                
                data.qvel[vel_idx] = safe_vx
                data.qvel[vel_idx + 1] = safe_vy
                
                # Dampen rotation heavily
                data.qvel[vel_idx + 3] *= 0.5
                data.qvel[vel_idx + 4] *= 0.5
                data.qvel[vel_idx + 5] *= 0.5
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
