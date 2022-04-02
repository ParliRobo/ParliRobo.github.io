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

![license](https://img.shields.io/badge/Platform-Android-green "Android&iOS")
![license](https://img.shields.io/badge/Version-Beta-yellow "Version")
![license](https://img.shields.io/badge/Licence-Apache%202.0-blue.svg "Apache")

## Table of Contents
[Introduction](#Introduction)

[Code Release](#code-release)


[Data Release](#data-release)

## Introduction
Recent years have witnessed the profound influence of AI technologies on computer gaming. While grandmaster-level AI robots have largely come true for complex games based on heavy back-end support, in practice many game developers crave for participant AI robots (PARs) that behave like average-level humans with inexpensive infrastructures. Unfortunately, to date there has not been a satisfactory solution that registers large-scale use. In this work, we attempt to develop practical PARs (called ParliRobo) showing acceptably humanoid behaviors with well affordable infrastructures under a challenging scenario—a complex 3D-FPS mobile game with real-time interaction requirements. Based on comprehensive explorations, we eventually enable this attempt through a novel “transform and polish” methodology, which achieves ultralight implementations of the core system components by non-intuitive yet principled approaches and meanwhile carefully fixes the probable side effect incurred on user perceptions. Evaluation results from large-scale deployment indicate the close resemblance in most biofidelity metrics between ParliRobo and human players; moreover, in 73% mini Turing tests ParliRobo cannot be distinguished from human players.

## Code Release

Currently, we are scrutinizing the codebase to avoid possible anonymity violation. To this end, we will release the source code of this study in a module-by-module manner as soon as we have finished examining a module and acquire its release permission from the authority.The codebase is organized as follows.
```
code
|---- agent
          |---- KLDiv.py
          |---- categorical.py
          |---- drtrace.py
          |---- entropy.py
          |---- get_shape.py
          |---- mmo.py
          |---- policy_graph.py
|---- _math.py
|---- preprocessor.py
```

+ `code/_math.py` provides math tools for clipping and embedding.
+ `code/preprocessor.py` preprocesses the visual information of AI robots, It preprocesses the visual information of AI robots and defines functions such as bilinear approximation and reward calculation.
+ `code/policy_graph.py` defines the structure of the Parlirobo's ML model.
+ `code/agent/policy_graph.py` is the entry file for the model.

The released part can be found [here](https://github.com/ParliRobo/ParliRobo.github.io/tree/main/code).

## Data Release
The video of the human player fighting with ParliRobo is as follows.

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


The data is being processed and will be released gradually.
