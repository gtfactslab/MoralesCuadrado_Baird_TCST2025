import os
import csv
import numpy as np

class Logger:

    def __init__(self, filename):
        self.filename = filename[0]
        # base_path = os.path.dirname(os.path.abspath(__file__))        # Get the directory where the script is located
        # base_path = os.path.join(base_path, 'data_analysis')

        base_path = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
        # base_path = os.path.abspath(os.path.join(base_path, ".."))  # A way to go up one level
        base_path = os.path.join(base_path, 'data_analysis/') # Append the 'data_analysis' folder to the path
        parts = base_path.split(os.sep)        # Split the path into components
        parts = ["src" if part == "build" else part for part in parts]        # Replace 'build' with 'src' if it exists in the path
        base_path = os.sep.join(parts)        # Reconstruct the new path

        print(f"logger {base_path = }")        # Print the base path
        self.filename = filename[0]        # Assuming 'filename' is passed or defined as a list
        self.full_path = os.path.join(base_path, self.filename)        # Combine the base path with the filename
        print(f"Logging to: {self.full_path}")        # Print the full path
        os.makedirs(os.path.dirname(self.full_path), exist_ok=True)        # Ensure the directory exists, and creates it if it doesn't


    def log(self, ControlNode):
        with open(self.full_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['time',
                            'x', 'y', 'z', 'yaw',
                            'throttle', 'roll_rate', 'pitch_rate', 'yaw_rate',
                            'x_ref', 'y_ref', 'z_ref', 'yaw_ref',
                            'pred_time', 'nr_time', 'ctrl_callback_time', 'metadata'
                            ])
            
            time_history = ControlNode.get_ctrl_loop_time_log() #0

            x_history = ControlNode.get_x_log() #1
            y_history = ControlNode.get_y_log() #2
            z_history = ControlNode.get_z_log() #3
            yaw_history = ControlNode.get_yaw_log() #4
            throttle_history = ControlNode.get_throttle_log() #5
            roll_history = ControlNode.get_roll_log() #6
            pitch_history = ControlNode.get_pitch_log() #7
            yaw_rate_history = ControlNode.get_yaw_rate_log() #8
            ref_x_history = ControlNode.get_ref_x_log() #9
            ref_y_history = ControlNode.get_ref_y_log() #10
            ref_z_history = ControlNode.get_ref_z_log() #11
            ref_yaw_history = ControlNode.get_ref_yaw_log() #12
            pred_time_history = ControlNode.get_pred_timel_log() #13
            nr_time_history = ControlNode.get_nr_timel_log() #14
            ctrl_callback_time_history = ControlNode.get_ctrl_callback_timel_log() #15
            metadata = ControlNode.get_metadata() #16
            """
            def get_x_log(self): return np.array(self.x_log).reshape(-1, 1)
            def get_y_log(self): return np.array(self.y_log).reshape(-1, 1)
            def get_z_log(self): return np.array(self.z_log).reshape(-1, 1)
            def get_yaw_log(self): return np.array(self.yaw_log).reshape(-1, 1)
            def get_throttle_log(self): return np.array(self.throttle_log).reshape(-1, 1)
            def get_roll_log(self): return np.array(self.roll_log).reshape(-1, 1)
            def get_pitch_log(self): return np.array(self.pitch_log).reshape(-1, 1)
            def get_yaw_rate_log(self): return np.array(self.yaw_rate_log).reshape(-1, 1)
            def get_ref_x_log(self): return np.array(self.ref_x_log).reshape(-1, 1)
            def get_ref_y_log(self): return np.array(self.ref_y_log).reshape(-1, 1)
            def get_ref_z_log(self): return np.array(self.ref_z_log).reshape(-1, 1)
            def get_ref_yaw_log(self): return np.array(self.ref_yaw_log).reshape(-1, 1)
            def get_ctrl_loop_time_log(self): return np.array(self.ctrl_loop_time_log).reshape(-1, 1)
            def get_pred_timel_log(self): return np.array(self.pred_timel_array).reshape(-1, 1)
            def get_nr_timel_log(self): return np.array(self.nr_timel_array).reshape(-1, 1)
            def get_ctrl_callback_timel_log(self): return np.array(self.ctrl_callback_timel_log).reshape(-1, 1)
            def get_metadata(self): return self.metadata.reshape(-1, 1)


            """
            
            # Pad the metadata to match the time history
            padding_length = time_history.shape[0] - metadata.shape[0]
            metadata = np.pad(metadata, ((0, padding_length), (0, 0)), 'constant', constant_values='0')
           

            # Combine the histories for logging
            data = np.hstack((time_history,
                              x_history, y_history, z_history, yaw_history,
                              throttle_history, roll_history, pitch_history, yaw_rate_history,
                              ref_x_history, ref_y_history, ref_z_history, ref_yaw_history,
                              pred_time_history, nr_time_history, ctrl_callback_time_history,
                              metadata
                              ))
            # Write each row to the CSV file
            for row in range(data.shape[0]):
                writer.writerow(np.asarray(data[row, :]).flatten())

            print(f"\nWrote to {self.full_path}")
