from mobilerack_pybind_module import QualisysClient
import time
from time import sleep
import pickle

"""
Stream frames & images from Qualisys Track Manager.
Images are not streamed in playback mode.

May need extra package in order to display images:
```bash
pip3 install opencv-python
```
"""
max_duration = 120
frame_list = []
qc = QualisysClient(3, [], "6D")

sleep(2)  # hacky way to wait until data is received from QTM
start = time.time()
current_duration = 0
while current_duration < max_duration:
    current_duration = time.time() - start
    print("Recording for", current_duration, "s")

    # Get frames
    frames, timestamp = qc.getData6D()

    # Get timestamp
    frames.append(time.time())

    frame_list.append(frames)
    sleep(0.01)

print("Saved", len(frame_list), "at", len(frame_list)/current_duration, "FPS")

with open("output/rigid_frame_data.pkl", "wb") as file:
    pickle.dump(frame_list, file)