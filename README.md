# Master's Thesis on Adaptive State Based Systems

This repository provides the reproducible, Python scripts for the experiments provided in the Master's Thesis named Adaptive State Based Systems, authored by Laxmikant Shrikant Baheti, as a part of the course "Systems Engineering and Engineering Management" in the Department of "Electrical Energy Engineering", supervised by Dipl.-Info. Detlef Arend. 

The experiments provided in the thesis can be reproduced by running the scripts in the directory "src/experiments". It is important to note that along with the provided code, the repository also provides some further essential elements as follows:

- The repository provides a dataset representing the double pendulum system in MLPro. This dataset is used to train an MLP in supervised learning to learn the state transition in experiment_2a.
- The trained MLP model, along with the training scenario from MLPro, is also included as part of the repository. This pre-trained model is used to demonstrate its use as a simulation model in an adaptive state transition function.
- Finally, the repository also provide some necessary refactored classes, in the directory "src/refactored" which were necessary to perform the experiments. 
