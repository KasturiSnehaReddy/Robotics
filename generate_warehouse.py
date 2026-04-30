import os
import mujoco
import mujoco.viewer

def generate_mujoco_xml():
    print("Generating Aesthetic Mujoco XML (Primitive Based)...")
    
    # ---------------- MATERIALS & ASSETS ----------------
    xml_assets = """
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0 0 0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.8 0.8 0.8" rgb2="0.75 0.75 0.75"/>
    <material name="floor_mat" texture="grid" texrepeat="20 20" texuniform="true" reflectance="0.05"/>
    
    <material name="mat_blue_rack" rgba="0.05 0.15 0.35 1" reflectance="0.2"/>
    <material name="mat_box_brown" rgba="0.76 0.6 0.42 1" reflectance="0.0"/>
    <material name="mat_pillar" rgba="0.85 0.8 0.75 1" reflectance="0.0"/>
    <material name="mat_wall" rgba="0.8 0.8 0.8 1" reflectance="0.0"/>
    <material name="mat_barrel" rgba="0.9 0.8 0.1 1" reflectance="0.3"/>
    <material name="mat_office" rgba="0.7 0.7 0.75 1" reflectance="0.1"/>
    <material name="mat_office_roof" rgba="0.8 0.8 0.85 1" reflectance="0.1"/>
  </asset>
"""

    # ---------------- GENERATE SHELVES & BOXES ----------------
    shelf_xml = ""
    # Rack unit params
    rack_w, rack_d, rack_h = 2.0, 1.0, 3.0
    thickness = 0.05
    levels = [0.2, 1.2, 2.2] # Z heights of shelves
    
    # Racks layout
    y_aisles = [-6, -2, 2, 6]
    x_positions = [-10, -6, -2, 2, 6] # End to end racks
    
    rack_counter = 0
    for y in y_aisles:
        for x in x_positions:
            rack_counter += 1
            # Vertical posts
            posts = ""
            for px in [-rack_w/2, rack_w/2]:
                for py in [-rack_d/2, rack_d/2]:
                    posts += f'<geom type="box" size="{thickness} {thickness} {rack_h/2}" pos="{px} {py} {rack_h/2}" material="mat_blue_rack"/>\n'
            
            # Horizontal levels & boxes
            shelves_and_boxes = ""
            for z in levels:
                # The shelf plank
                shelves_and_boxes += f'<geom type="box" size="{rack_w/2} {rack_d/2} {thickness}" pos="0 0 {z}" material="mat_blue_rack"/>\n'
                
                # Boxes on this shelf (let's put 4 boxes per level)
                box_size = 0.35
                for bx in [-0.75, -0.25, 0.25, 0.75]:
                    shelves_and_boxes += f'<geom type="box" size="{box_size/2} {box_size/2} {box_size/2}" pos="{bx} 0 {z + thickness + box_size/2}" material="mat_box_brown"/>\n'

            shelf_xml += f"""
    <!-- Rack {rack_counter} -->
    <body name="rack_{rack_counter}" pos="{x} {y} 0">
      {posts}
      {shelves_and_boxes}
    </body>"""

    # ---------------- WALLS ----------------
    walls_xml = """
    <!-- Perimeter Walls -->
    <body name="wall_left" pos="-16 0 2.5">
      <geom type="box" size="0.2 11 2.5" material="mat_wall"/>
    </body>
    <body name="wall_right" pos="16 0 2.5">
      <geom type="box" size="0.2 11 2.5" material="mat_wall"/>
    </body>
    <body name="wall_top" pos="0 11 2.5">
      <geom type="box" size="16.2 0.2 2.5" material="mat_wall"/>
    </body>
    <body name="wall_bottom" pos="0 -11 2.5">
      <geom type="box" size="16.2 0.2 2.5" material="mat_wall"/>
    </body>"""

    # ---------------- PILLARS ----------------
    pillars_xml = ""
    pillar_x = [-14, 0, 14]
    pillar_y = [-9.5, 9.5]
    p_idx = 0
    for px in pillar_x:
        for py in pillar_y:
            p_idx += 1
            pillars_xml += f"""
    <body name="pillar_{p_idx}" pos="{px} {py} 2.5">
      <geom type="box" size="0.3 0.3 2.5" material="mat_pillar"/>
    </body>"""

    # ---------------- BARRELS ----------------
    barrels_xml = ""
    b_idx = 0
    for bx in [12, 13, 14]:
        for by in [7, 8]:
            b_idx += 1
            barrels_xml += f"""
    <body name="barrel_{b_idx}" pos="{bx} {by} 0.5">
      <geom type="cylinder" size="0.4 0.5" material="mat_barrel"/>
    </body>"""

    # ---------------- OFFICE BLOCK ----------------
    office_xml = """
    <!-- Office Block -->
    <body name="office_container" pos="12 -7 1.5">
      <geom type="box" size="3 2 1.5" material="mat_office"/>
      <geom type="box" size="3.1 2.1 0.1" pos="0 0 1.5" material="mat_office_roof"/>
    </body>"""

    # ---------------- ROBOTS ----------------
    import os
    is_exploration = not os.path.exists("scanned_grid.npy")
    
    robots_xml = """
    <!-- Robot 1 (The Mapper) -->
    <body name="robot1" pos="-12 -8 0.2">
      <freejoint/>
      <geom type="cylinder" size="0.3 0.1" rgba="0.2 0.8 0.2 1" mass="10.0"/>
      <geom type="cylinder" size="0.1 0.05" pos="0 0 0.15" rgba="0.1 0.1 0.1 1" mass="1.0"/>
      <camera name="lidar_cam1" pos="0 0.25 0.15" euler="90 0 0" fovy="90"/>
    </body>"""

    if not is_exploration:
        robots_xml += """
    <!-- Robot 2 -->
    <body name="robot2" pos="-12 8 0.2">
      <freejoint/>
      <geom type="cylinder" size="0.3 0.1" rgba="0.8 0.2 0.2 1" mass="10.0"/>
      <geom type="cylinder" size="0.1 0.05" pos="0 0 0.15" rgba="0.1 0.1 0.1 1" mass="1.0"/>
      <camera name="lidar_cam2" pos="0 0.25 0.15" euler="90 0 0" fovy="90"/>
    </body>"""

    # ---------------- FULL XML ASSEMBLE ----------------
    xml_string = f"""<mujoco model="warehouse">
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <compiler angle="degree"/>
  {xml_assets}
  <worldbody>
    <!-- Lighting -->
    <light pos="0 0 15" dir="0 0 -1" directional="true" diffuse="0.9 0.9 0.9" specular="0.1 0.1 0.1" castshadow="true"/>
    <light pos="-10 0 10" dir="1 0 -1" directional="false" diffuse="0.4 0.4 0.4"/>
    <light pos="10 0 10" dir="-1 0 -1" directional="false" diffuse="0.4 0.4 0.4"/>
    
    <!-- Floor -->
    <geom type="plane" size="16 11 0.05" material="floor_mat"/>
    
    {walls_xml}
    {pillars_xml}
    {office_xml}
    {barrels_xml}
    {shelf_xml}
    {robots_xml}
    
  </worldbody>
</mujoco>"""

    with open("warehouse.xml", "w") as f:
        f.write(xml_string)
    print("Saved beautiful warehouse to warehouse.xml")
    return xml_string

def main():
    generate_mujoco_xml()
    
    print("Launching Mujoco Viewer...")
    model = mujoco.MjModel.from_xml_path("warehouse.xml")
    data = mujoco.MjData(model)
    
    # Customize viewer to look down like the reference image
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -45
        viewer.cam.distance = 40.0
        viewer.cam.lookat[:] = [0, 0, 0]
        
        print("Viewer running. Close window to exit.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()
