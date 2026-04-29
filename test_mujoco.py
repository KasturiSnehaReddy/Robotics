import mujoco
import mujoco.viewer
import time

# Define a simple environment with a floor and a falling box
xml_string = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
  </asset>
  
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" directional="true"/>
    <geom type="plane" size="0 0 0.05" material="grid"/>
    
    <body pos="0 0 2" name="box">
      <freejoint/>
      <geom type="box" size="0.2 0.2 0.2" rgba="1 0 0 1" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""

def main():
    print("Compiling Mujoco XML model...")
    # Load the model and data
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)

    print("Launching Mujoco interactive viewer...")
    # Launch the viewer and run the simulation
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        # Run simulation for a while
        start_time = time.time()
        while viewer.is_running() and time.time() - start_time < 10:
            step_start = time.time()
            
            # Step the physics
            mujoco.mj_step(model, data)
            
            # Update the viewer
            viewer.sync()

            # Timekeeping to match real-time
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                
    print("Simulation complete.")

if __name__ == "__main__":
    main()
