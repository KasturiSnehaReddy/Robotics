import time
import math
import mujoco
import mujoco.viewer
import networkx as nx
from path_planning import build_warehouse_graph, euclidean_distance

def main():
    print("=== Step 1: Brain (Path Planning) ===")
    G, obstacles = build_warehouse_graph()
    start_node = (0, -8)
    goal_node = (3, 8)
    
    try:
        path = nx.astar_path(G, source=start_node, target=goal_node, heuristic=euclidean_distance)
        print(f"Path calculated! {len(path)} waypoints.")
    except nx.NetworkXNoPath:
        print("ERROR: No path found!")
        return

    # Convert path to a list of floating point targets we can pop from
    waypoints = [(float(p[0]), float(p[1])) for p in path]
    
    print("=== Step 2: Body (Mujoco Physics) ===")
    model = mujoco.MjModel.from_xml_path("warehouse.xml")
    data = mujoco.MjData(model)
    
    # We will use a simple Proportional (P) velocity controller
    # Target velocity speed in m/s
    MAX_SPEED = 2.0 
    
    print("Launching Viewer. Watch the robot go!")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            
            # --- Controller Logic ---
            if len(waypoints) > 0:
                target_x, target_y = waypoints[0]
                
                # Get current robot position
                # Robot is a freejoint, its qpos index starts at 0
                current_x = data.qpos[0]
                current_y = data.qpos[1]
                
                # Distance to current waypoint
                dist = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
                
                if dist < 0.2:
                    # We reached the waypoint! Pop it and target the next one.
                    waypoints.pop(0)
                    if len(waypoints) == 0:
                        print("Goal Reached!")
                        # Stop the robot
                        data.qvel[0] = 0
                        data.qvel[1] = 0
                else:
                    # Calculate velocity vector towards waypoint
                    dir_x = (target_x - current_x) / dist
                    dir_y = (target_y - current_y) / dist
                    
                    # Set the linear velocity of the freejoint
                    data.qvel[0] = dir_x * MAX_SPEED
                    data.qvel[1] = dir_y * MAX_SPEED
                    
                    # Dampen the rotation so it doesn't spin wildly or tip over
                    data.qvel[3] *= 0.5  # wx
                    data.qvel[4] *= 0.5  # wy
                    data.qvel[5] *= 0.5  # wz
            
            # --- Physics Step ---
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Maintain real-time
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
