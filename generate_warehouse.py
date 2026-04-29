import os
import trimesh
from shapely.geometry import Polygon, box
import mujoco
import mujoco.viewer

def create_walls_mesh():
    print("Generating Warehouse Walls (Shapely + Trimesh)...")
    # Outer 20x20, inner 19x19
    outer_wall = box(-10, -10, 10, 10)
    inner_empty_space = box(-9.5, -9.5, 9.5, 9.5)
    walls_2d = outer_wall.difference(inner_empty_space)
    
    # Extrude
    if walls_2d.geom_type == 'MultiPolygon':
        meshes = [trimesh.creation.extrude_polygon(p, height=3.0) for p in walls_2d.geoms]
        walls_mesh = trimesh.util.concatenate(meshes)
    else:
        walls_mesh = trimesh.creation.extrude_polygon(walls_2d, height=3.0)
        
    os.makedirs('assets', exist_ok=True)
    walls_path = os.path.join('assets', 'walls.obj')
    walls_mesh.export(walls_path)
    print(f"Saved walls to {walls_path}")

def create_shelf_mesh():
    print("Generating Detailed Shelf Rack (Trimesh)...")
    # Dimensions: 4m long, 1m deep, 2.5m tall
    w, d, h = 4.0, 1.0, 2.5
    thickness = 0.1
    
    meshes = []
    # 4 Vertical posts
    for x in [-w/2 + thickness/2, w/2 - thickness/2]:
        for y in [-d/2 + thickness/2, d/2 - thickness/2]:
            post = trimesh.creation.box(extents=[thickness, thickness, h])
            post.apply_translation([x, y, h/2])
            meshes.append(post)
            
    # 4 Horizontal levels
    for z in [0.2, 0.9, 1.6, 2.3]:
        level = trimesh.creation.box(extents=[w, d, thickness])
        level.apply_translation([0, 0, z])
        meshes.append(level)
        
    shelf_mesh = trimesh.util.concatenate(meshes)
    shelf_path = os.path.join('assets', 'shelf.obj')
    shelf_mesh.export(shelf_path)
    print(f"Saved shelf to {shelf_path}")

def generate_mujoco_xml():
    print("Generating Aesthetic Mujoco XML...")
    
    # Generate shelf placements
    # 3 rows of shelves, each row has 2 units placed end-to-end
    shelf_xml = ""
    y_positions = [-4, 0, 4] # Aisles
    x_positions = [-3, 3]    # End to end
    
    for y in y_positions:
        for x in x_positions:
            shelf_xml += f"""
    <body name="shelf_{x}_{y}" pos="{x} {y} 0">
      <geom type="mesh" mesh="shelf_mesh" material="rack_orange" contype="1" conaffinity="1"/>
    </body>"""

    xml_string = f"""<mujoco model="warehouse">
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <compiler angle="degree"/>
  
  <asset>
    <!-- Premium Textures -->
    <texture type="skybox" builtin="gradient" rgb1="0.1 0.15 0.2" rgb2="0 0 0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.15 0.15 0.15" rgb2="0.2 0.2 0.2"/>
    <material name="floor_mat" texture="grid" texrepeat="10 10" texuniform="true" reflectance="0.1"/>
    
    <material name="wall_concrete" rgba="0.7 0.7 0.7 1" reflectance="0.0"/>
    <material name="rack_orange" rgba="1.0 0.4 0.0 1" reflectance="0.2"/>
    <material name="dock_yellow" rgba="0.9 0.8 0.1 1" reflectance="0.1"/>
    
    <!-- Load Meshes -->
    <mesh name="walls_mesh" file="assets/walls.obj"/>
    <mesh name="shelf_mesh" file="assets/shelf.obj"/>
  </asset>
  
  <worldbody>
    <!-- Lighting -->
    <light pos="0 0 10" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>
    <light pos="-8 0 5" dir="1 0 -1" directional="false" diffuse="0.5 0.5 0.5"/>
    <light pos="8 0 5" dir="-1 0 -1" directional="false" diffuse="0.5 0.5 0.5"/>
    
    <!-- The Ground -->
    <geom type="plane" size="15 15 0.05" material="floor_mat"/>
    
    <!-- The Warehouse Walls -->
    <body name="warehouse_walls" pos="0 0 0">
      <geom type="mesh" mesh="walls_mesh" material="wall_concrete" contype="1" conaffinity="1"/>
    </body>
    
    <!-- Shelving Units -->{shelf_xml}
    
    <!-- Loading Dock Area -->
    <body name="loading_dock" pos="0 -9 0.1">
      <geom type="box" size="8 1 0.1" material="dock_yellow"/>
    </body>
    
    <!-- Robot 1 (Blue) -->
    <body name="robot1" pos="0 -8 0.5">
      <freejoint/>
      <!-- Chassis -->
      <geom type="cylinder" size="0.25 0.1" rgba="0.2 0.6 1.0 1" mass="10.0"/>
      <!-- LiDAR dome -->
      <geom type="cylinder" size="0.1 0.05" pos="0 0 0.15" rgba="0.1 0.1 0.1 1" mass="1.0"/>
    </body>
    
    <!-- Robot 2 (Red) -->
    <body name="robot2" pos="0 8 0.5">
      <freejoint/>
      <geom type="cylinder" size="0.25 0.1" rgba="1.0 0.2 0.2 1" mass="10.0"/>
      <geom type="cylinder" size="0.1 0.05" pos="0 0 0.15" rgba="0.1 0.1 0.1 1" mass="1.0"/>
    </body>
    
  </worldbody>
</mujoco>
"""
    with open("warehouse.xml", "w") as f:
        f.write(xml_string)
    print("Saved Mujoco configuration to warehouse.xml")
    return xml_string

def main():
    create_walls_mesh()
    create_shelf_mesh()
    xml_string = generate_mujoco_xml()
    
    print("Launching Mujoco Viewer...")
    model = mujoco.MjModel.from_xml_path("warehouse.xml")
    data = mujoco.MjData(model)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer running. Close window to exit.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()
