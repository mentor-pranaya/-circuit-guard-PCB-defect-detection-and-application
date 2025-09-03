# create_dirs.py
import os
os.makedirs('outputs_module2/bbox', exist_ok=True)
os.makedirs('outputs_module2/rois/unlabeled', exist_ok=True)
print("created outputs_module2/bbox and outputs_module2/rois/unlabeled")
