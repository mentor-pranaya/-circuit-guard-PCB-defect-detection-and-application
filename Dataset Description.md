Dataset Description



Overview



This dataset consists of Printed Circuit Board (PCB) images collected for the purpose of defect detection and classification. PCBs are critical components in electronic devices, and even minor manufacturing defects can lead to product malfunction. This dataset provides both defect-free reference images and defective PCB samples across multiple defect categories.



It is suitable for research in image processing, computer vision, and deep learning, with applications in automated quality control systems in electronics manufacturing.



Dataset Structure



The dataset is organized into directories based on defect type. Each folder contains images of PCBs belonging to that category.



PCB\_DATASET/

│

├── images/

│   ├── Missing\_hole/

│   ├── Mouse\_bite/

│   ├── Open\_circuit/

│   ├── Short/

│   ├── Spur/

│   ├── Spurious\_copper/

│

├──Annnotations/

│   ├── Missing\_hole/

│   ├── Mouse\_bite/

│   ├── Open\_circuit/

│   ├── Short/

│   ├── Spur/

│   ├── Spurious\_copper/

│

├──PCB Used/

├──rotation/

├── results/

│   ├── difference/

│   ├── defect\_masks/

│   └── highlighted/

├── contours.ipynb/

├── subtraction.ipynb/



### **Dataset Contents**



    Raw PCB Images (images/)

    Contains PCB images categorized into 6 defect types:



        Missing\_hole → Missing drilled holes on PCB.



        Mouse\_bite → Small notches or irregular edges.



        Open\_circuit → Broken traces causing open connections.



        Short → Unwanted solder bridges/trace connections.



        Spur → Extra thin copper lines.



        Spurious\_copper → Random unwanted copper deposits.



    Annotations (Annotations/)

    Ground-truth files (masks or labels) corresponding to each defect category.



    Reference PCB (PCB\_Used/)

    Contains ideal PCB images without defects used as baselines for subtraction.



    Rotation (rotation/)

    Contains rotated versions of PCB images for data augmentation and alignment.



    Results (results/)



        difference/ → Image subtraction outputs (reference vs. test PCB).



        defect\_masks/ → Binary masks highlighting defect regions.



        highlighted/ → Visual overlays showing defects on PCBs.



    Notebooks



        subtraction.ipynb → Code for background subtraction \& defect detection.



        contours.ipynb → Code for extracting and visualizing defect contours.

