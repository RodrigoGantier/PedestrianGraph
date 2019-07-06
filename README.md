# Pedestrian Graph: Pedestrian Crossing Prediction Base on 2D Pose Estimation and Graph Convolutional Networks

This project hosts the code for implementing the Pedestrian Graph algorithm for crossing prediction, as presented in our paper:

    Pedestrian Graph: Pedestrian Crossing Prediction Base on 2D Pose Estimation and Graph Convolutional Networks;
    Pablo Rodrigo Gantier Cadena, Ming Yang, Yeqiang Qian and Chunxiang Wang;

The full paper is available at: soon. 

## Highlights
- **State-of-the-art:** In the Joint Attention in Autonomous Driving (JAAD) data set 91.94% accuracy.   
- **Speed:** Each sample take 0.02328 sec.
- **Contribution:** The development of the 2D Pedestrian Graph structure and Pedestrian Graph Network to predict whether a pedestrian is going to cross the street.
- **Fast training:** Less than 1 hour for training (test in 2 gtx 1080).

## Required hardware
We use 2 Nvidia GTX1080 GPUs. 

## Installation
Needs:

python3 

Tensorflow 1.13 or higher.
 
Opencv 3.3 or higher 

xml.etree

glob 

random

7z (sudo apt install p7zip-full)

Tested on ubuntu 16.04 and 18.04

## Reproduce results
Once all requirements are install, you can follow the below steps to run a quick evaluation.
    
    # Clone the repositpry 
    git clone https://github.com/RodrigoGantier/PedestrianGraph.git && cd PedestrianGraph
    7z e tfrecords.7z -o./tfrecords/
    python3 train_eval.py -c tfrecords --train False
    
    # tfrecod files are provided for a quick evaluation


## To generate tfrecord files was use to_tfrecord.py
The command are as follow:
    
    7z e no_cro2d.7z -o./no_cro2d/
    7z e will_cro2d.7z -o./will_cro2d/
    python3 to_tfrecord.py -c ./
    # to train just:
    python3 train_eval.py -c tfrecords --train True --num_gpu 2
    # to visualize tensorboard 
    tensorboard --logdir=<to your path>    

## Visualize the 14 extracted key points
The command are as follow:
    
    curl -o JAAD_clips.7z https://drive.google.com/file/d/1RdqSptdttunn1uFd4QcscuYTcmSXKuMd/view?usp=sharing
    7z e skeleton_cpn.7z -o./skeleton_cpn/
    7z e JAAD_clips.7z -o./JAAD_clips/
    python3 visual_keypoints.py -c ./JAAD_clips -i ./skeleton_cpn  


## Citations
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.
```
@article{Gantier2019PedestrianGraph,
  title   =  {{Pedestrian Graph}: Pedestrian Crossing Prediction Base on 2D Pose Estimation and Graph Convolutional Networks},
  author  =  {Pablo Rodrigo Gantier Cadena, Ming Yang, Yeqiang Qian and Chunxiang Wang},
  journal =  {ITSC2019},
  year    =  {2019}
}
```


## License

For academic use. For commercial use, please contact the authors. 

