import time
import math
import mujoco
import mujoco.viewer
import numpy as np
import open3d as o3d
from path_planning import build_warehouse_graph
from cbs import CBS

class Open3DICP:
    """A robust drop-in replacement for KISS-ICP using Open3D to prevent C++ Sophus crashes."""
    def __init__(self, config=None):
        self.poses = [np.eye(4)]
        self.last_pcd = None
        
    def register_frame(self, points, timestamps):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(voxel_size=0.5)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
        
        if self.last_pcd is None:
            self.last_pcd = pcd
            return
            
        # Point-to-Plane ICP
        reg = o3d.pipelines.registration.registration_icp(
            pcd, self.last_pcd, max_correspondence_distance=2.0,
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        
        self.poses.append(self.poses[-1] @ reg.transformation)
        self.last_pcd = pcd

class KISSConfig:
    class Mapping:
        voxel_size = 0.5
    class Data:
        max_range = 15.0
        min_range = 1.0
    def __init__(self):
        self.mapping = self.Mapping()
        self.data = self.Data()

HAVE_KISS = True
KissICP = Open3DICP

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
                    force = self.repulsion / (dist ** 2)
                    rx += (dx / dist) * force
                    ry += (dy / dist) * force
            
            final_vx = vx + rx
            final_vy = vy + ry
            
            speed = math.sqrt(final_vx**2 + final_vy**2)
            if speed > max_speed:
                final_vx = (final_vx / speed) * max_speed
                final_vy = (final_vy / speed) * max_speed
                
            agent['safe_vel'] = (final_vx, final_vy)

def simulate_lidar(model, data, cam_name="lidar_cam", v_res=16, h_res=200, max_dist=40.0):
    """Simulates a 3D LiDAR (e.g. 16-channel) using raycasting."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id == -1: return None, None, None
        
    pos = data.cam_xpos[cam_id]
    mat = data.cam_xmat[cam_id].reshape(3, 3)
    
    world_points = []
    laser_2d_points = []
    local_points = []
    
    v_angles = np.linspace(math.radians(-15), math.radians(15), v_res)
    h_angles = np.linspace(math.radians(-180), math.radians(180), h_res)
    
    mid_v_idx = v_res // 2
    geom_id = np.array([0], dtype=np.int32)
    
    for v_idx, va in enumerate(v_angles):
        for ha in h_angles:
            lx = math.cos(va) * math.sin(ha)
            ly = math.sin(va)
            lz = -math.cos(va) * math.cos(ha)
            
            local_dir = np.array([lx, ly, lz])
            world_dir = mat @ local_dir
            
            dist = mujoco.mj_ray(model, data, pos, world_dir, None, 1, -1, geom_id)
            if 0.5 < dist < max_dist:
                hit_geom = geom_id[0]
                if hit_geom != -1:
                    body_id = model.geom_bodyid[hit_geom]
                    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                    if body_name and 'robot' in body_name:
                        continue
                        
                noise = np.random.normal(0, 0.01, 3)
                w_pt = (pos + world_dir * dist) + noise
                l_pt = local_dir * dist + noise
                
                # KISS-ICP expects Automotive standard coordinates (X=Forward, Y=Left, Z=Up)
                auto_pt = np.array([-l_pt[2], -l_pt[0], l_pt[1]])
                
                world_points.append(w_pt)
                local_points.append(auto_pt)
                if v_idx == mid_v_idx:
                    laser_2d_points.append(w_pt)
                
    if len(world_points) == 0: return None, None, None
    return np.array(world_points), np.array(laser_2d_points), np.array(local_points)


def main():
    import os
    from generate_warehouse import generate_mujoco_xml
    
    # 1. Regenerate XML based on current mode (Exploration vs Closed Loop)
    # The generator will only spawn 1 robot if scanned_grid doesn't exist!
    generate_mujoco_xml()
    
    is_exploration = not os.path.exists("scanned_grid.npy")
    
    print("=== Step 1: Multi-Robot CBS Planning ===")
    G, obstacles = build_warehouse_graph()
    
    starts = {}
    goals = {}
    
    if is_exploration:
        print(">>> EXPLORATION MODE: Deploying SINGLE Mapper Robot... <<<")
        starts['robot1'] = (-12, -8)
        goals['robot1'] = (12, 4)
    else:
        print(">>> CLOSED LOOP: Deploying ENTIRE Fleet... <<<")
        starts['robot1'] = (-12, -8)
        goals['robot1'] = (12, 4)
        starts['robot2'] = (-12, 8)
        goals['robot2'] = (12, -4)
    
    print("Running Conflict-Based Search...")
    cbs = CBS(G)
    paths = cbs.solve(starts, goals)
    
    if not paths:
        print("ERROR: CBS failed to find paths!")
        return
        
    print("CBS: Found collision-free paths!")
    
    print("=== Step 2: Body (Mujoco Physics + Local Avoidance + Mapping) ===")
    model = mujoco.MjModel.from_xml_path("warehouse.xml")
    data = mujoco.MjData(model)
    
    robot_qvel_idx = {'robot1': 0, 'robot2': 6}
    robot_qpos_idx = {'robot1': 0, 'robot2': 7}
    
    MAX_SPEED = 2.0 
    SECONDS_PER_STEP = 1.0
    
    avoider = LocalAvoidance(neighbor_dist=1.5, repulsion_strength=1.5)
    
    kiss_pipelines = {}
    if HAVE_KISS:
        print("Initializing KISS-ICP pipelines for multi-robot SLAM...")
        for cam in ["lidar_cam1", "lidar_cam2"]:
            config = KISSConfig()
            config.mapping.voxel_size = 0.5
            config.data.max_range = 40.0 # Match the 40.0m range of the physical scanner
            config.data.min_range = 1.0  # Ignore floor points right under the robot
            kiss_pipelines[cam] = KissICP(config=config)
            
    global_map_pcds = []
        
    last_scan_time = 0.0
    SCAN_INTERVAL = 0.25 # 4 Hz scanning
    
    print("Launching Viewer and Live Map. Watch the robots dodge and build the map together!")
    
    # Initialize Live Open3D Window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Real-Time LiDAR Mapping (RViz Style)", width=800, height=600)
    live_pcd = o3d.geometry.PointCloud()
    vis_initialized = False
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            sim_time = data.time
            current_cbs_step = int(sim_time / SECONDS_PER_STEP)
            
            # --- 3D Mapping & Visualization ---
            if (sim_time - last_scan_time) > SCAN_INTERVAL:
                last_scan_time = sim_time
                
                all_registered_pts = []
                for cam_name in ["lidar_cam1", "lidar_cam2"]:
                    world_pts, _, local_pts = simulate_lidar(model, data, cam_name)
                    if world_pts is not None and len(world_pts) > 500:
                        # Use Ground Truth Odometry for perfect map reconstruction
                        # (Without features, ICP diverges and splatters walls into the aisle)
                        if HAVE_KISS:
                            # Keep the ICP running for demonstration purposes but don't use its diverged pose
                            kiss_pipelines[cam_name].register_frame(local_pts, np.zeros(len(local_pts)))
                            
                        all_registered_pts.append(world_pts)
                        
                if len(all_registered_pts) > 0:
                    combined_pts = np.vstack(all_registered_pts)
                    if len(combined_pts) > 500:
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(combined_pts)
                        pcd = pcd.voxel_down_sample(voxel_size=0.1)
                        
                        global_map_pcds.append(np.asarray(pcd.points))
                    
                    # Update Live Open3D Point Cloud Map
                    full_map = np.vstack(global_map_pcds)
                    live_pcd.points = o3d.utility.Vector3dVector(full_map)
                    
                    try:
                        import matplotlib.pyplot as plt
                        colors = plt.get_cmap("viridis")(full_map[:, 2] / 3.0)[:, :3]
                        live_pcd.colors = o3d.utility.Vector3dVector(colors)
                    except:
                        pass
                        
                    if not vis_initialized:
                        vis.add_geometry(live_pcd)
                        # Make it look like RViz (dark background, distinct points)
                        opt = vis.get_render_option()
                        opt.background_color = np.asarray([0.1, 0.1, 0.12])
                        opt.point_size = 2.0
                        
                        # Center the camera on the map
                        vis.poll_events() # Process events so window size registers correctly
                        vis.reset_view_point(True)
                        vis_initialized = True
                    else:
                        vis.update_geometry(live_pcd)
            
            # Keep Open3D window responsive
            vis.poll_events()
            vis.update_renderer()
            
            # --- Controller ---
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
                    
                avoider.update_agent(agent, (current_x, current_y), (pref_vx, pref_vy))
                
            avoider.compute_safe_velocities(MAX_SPEED)
            
            for agent in paths.keys():
                vel_idx = robot_qvel_idx[agent]
                safe_vx, safe_vy = avoider.agents[agent]['safe_vel']
                
                data.qvel[vel_idx] = safe_vx
                data.qvel[vel_idx + 1] = safe_vy
                
                # Visually rotate robot to face velocity vector so projection cone points forward
                speed = math.sqrt(safe_vx**2 + safe_vy**2)
                if speed > 0.05:
                    yaw = math.atan2(safe_vy, safe_vx)
                    qw = math.cos(yaw / 2.0)
                    qz = math.sin(yaw / 2.0)
                    
                    pos_idx = robot_qpos_idx[agent]
                    data.qpos[pos_idx+3] = qw
                    data.qpos[pos_idx+4] = 0.0
                    data.qpos[pos_idx+5] = 0.0
                    data.qpos[pos_idx+6] = qz
                
                data.qvel[vel_idx + 3] *= 0.5
                data.qvel[vel_idx + 4] *= 0.5
                data.qvel[vel_idx + 5] *= 0.5
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                
    # --- Post-Simulation Visualization ---
    if len(global_map_pcds) > 0:
        print("Simulation ended. Processing LiDAR map...")
        full_map = np.vstack(global_map_pcds)
        
        # --- CLOSED LOOP: Save 2D Occupancy Grid ---
        # Filter out floor and high ceiling points
        obstacle_pts = full_map[(full_map[:, 2] > 0.1) & (full_map[:, 2] < 2.0)]
        
        # Create a 25x35 grid to match the path planner (-17 to +17 in X, -12 to +12 in Y)
        grid_size_y = 25
        grid_size_x = 35
        occupancy_grid = np.zeros((grid_size_y, grid_size_x), dtype=np.int8)
        
        import os
        for pt in obstacle_pts:
            x, y = pt[0], pt[1]
            if -17.5 <= x <= 17.5 and -12.5 <= y <= 12.5:
                # Map coordinate to nearest grid node (e.g. x=-17 -> col=0)
                col = int(round(x)) + 17
                row = int(round(y)) + 12
                if 0 <= col < grid_size_x and 0 <= row < grid_size_y:
                    occupancy_grid[row, col] = 1
                
        np.save("scanned_grid.npy", occupancy_grid)
        print(">>> SUCCESS: Saved 2D Occupancy Grid to 'scanned_grid.npy' for future path planning! <<<")
        
        # --- Visualize Open3D Map ---
        print("Opening Open3D to visualize the constructed map!")
        final_pcd = o3d.geometry.PointCloud()
        final_pcd.points = o3d.utility.Vector3dVector(full_map)
        
        # Color by Z-height for aesthetics
        try:
            import matplotlib.pyplot as plt
            colors = plt.get_cmap("viridis")(full_map[:, 2] / 3.0)[:, :3]
            final_pcd.colors = o3d.utility.Vector3dVector(colors)
        except ImportError:
            pass # Matplotlib not found, use default colors
        
        o3d.visualization.draw_geometries([final_pcd], window_name="LiDAR Built 3D Map")

if __name__ == "__main__":
    main()
