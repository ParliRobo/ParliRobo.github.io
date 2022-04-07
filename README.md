<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>
<br />

![license](https://img.shields.io/badge/Platform-Linux-green "Linux")
![license](https://img.shields.io/badge/Version-Beta-yellow "Version")
![license](https://img.shields.io/badge/Licence-Apache%202.0-blue.svg "Apache")

## Table of Contents
[Introduction](#Introduction)

[Implementation of ParliRobo](#implementation-of-parlirobo)

[Data Release](#data-release)

[Demo Video](#demo-video)

[Platform Requirements](#platform-requirements)

[For Developers](#for-developers)

## Introduction
Our work focuses on developing participant lightweight AI robots (PARs) for complex real-time mobile gaming. To this end, we collaborate with X-Game, a popular 3D-FPS real-time mobile game that owns ~210,000 users in 2021. Based on comprehensive explorations, we eventually propose a practical PAR system (called ParliRobo) through a novel “transform and polish” methodology, which achieves ultralight implementations of the core system components by non-intuitive yet principled approaches and meanwhile carefully fixes the probable side effect incurred on user perceptions. This repository contains the implementation code of ParliRobo (including the training code and deployment code) and our released data.

## Implementation of ParliRobo

Currently, we are scrutinizing the codebase to avoid possible anonymity violation. To this end, we will release the source code of this study in a module-by-module manner as soon as we have finished examining a module and acquire its release permission from the authority. The codebase is organized as follows.
```
code
|---- math.py
|---- preprocessor.py
|---- training
          |---- KLDiv.py
          |---- categorical.py
          |---- drtrace.py
          |---- entropy.py
          |---- get_shape.py
          |---- mmo.py
          |---- policy_graph.py
|---- deployment
          |---- client.py
          |---- server.py
          |---- GamebotAPI.py
          |---- pressure_test.py
          |---- preprocessor.py
```

+ `implementation/math.py` provides math tools for data clipping and embedding.
+ `implementation/preprocessor.py` preprocesses the visual information of participant AI robots by defining functions such as bilinear approximation.
+ `implementation/training` contains several modules for training PARs of ParliRobo.
+ `implementation/deployment` includes several modules for deploying ParliRobo.

The released part can be found [here](https://github.com/ParliRobo/ParliRobo.github.io/tree/main/code).

## Data Release
Currently, we have released a portion of the representative sample data (with proper anonymization) for references [here](https://github.com/ParliRobo/ParliRobo.github.io/tree/main/sample_dataset). As to the full dataset, we are still in discussion with the authority to what extent can it be release. We will make the rest dataset in public as soon as possible after receiving permissions and desensitizing the dataset.

These data are organized in interaction_delay.xlsx, mini-Turing_Test.xls, real-time_gaming_information.xlsx, respectively (for detailed data, please click [here](https://github.com/ParliRobo/ParliRobo.github.io/tree/main/sample_dataset). For each file, we list the specific information coupled with the regarding description as follows.
```
sample-dataset
|---- interaction_delay.xlsx
|---- real-time_gaming_information.xlsx
|---- mini-Turing_Test.xls
```

### interaction_delay.xlsx

| Column Name                | Description                                                  |
| -------------------------- | ------------------------------------------------------------ |
| `instruction_id`           | Unique ID generated to identify an instruction               |
| `system_name`              | PAR system name                                              |
| `interaction_delay`        | Average delay of instructions in a second                    |

### real-time_gaming_information.xlsx

| Column Name                | Description                                                  |
| -------------------------- | ------------------------------------------------------------ |
| `uid`                      | Unique ID generated to identify a PAR or human player (cannot be related to the user’s true indentity) |
| `solution_name`            | PAR systems or human player                                  |
| `shoots`                   | Number of shoots performed by a user                         |
| `hits`                     | Number of hits performed by a user                           |
| `jumps`                    | Number of jumps performed by a user                          |
| `crouches`                 | Number of crouches performed by a user                       |
| `moving distances`         | Length of moving distances performed by a user               |
| `pros gathered`            | Number of pros gathered performed by a user                  |

### mini-Turing_Test.xls

| Column Name                | Description                                                  |
| -------------------------- | ------------------------------------------------------------ |
| `video_id`                 | Unique ID generated to identify a game video clip            |
| `solution_name`            | PAR system or human player                                   |
| `V1` - `V34`               | Unique ID generated to identify a volunteer                  |

## Demo Video
We have released a demo video of a participant AI robot (PAR) trained by ParliRobo fighting with a human player. Note that the protagonist in this video is the PAR.

<iframe 
src="https://www.youtube.com/embed/Hw3-9WkHHpg" 
scrolling="no" 
border="0" 
frameborder="no" 
framespacing="0" 
allowfullscreen="true" 
height=300 
width=400> 
</iframe>

## Platform Requirements
Currently, we build and deploy ParliRobo on Linux servers with proper Python supports. Specifically, we leverage [TensorFlow](https://opensource.google/projects/tensorflow), an end-to-end open source platform for machine learning, to train participant AI robots (PARs) in this study.


## For Developers
Our code is licensed under Apache 2.0. Please adhere to the corresponding open source policy when applying modifications and commercial uses.
Also, some of our code is currently not available but will be relased soon once we have obatained permissions.
