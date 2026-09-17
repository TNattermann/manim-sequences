# manim

## Disclaimer 

This is the mirrored version of the AALAB research group repo at RPTU Kaiserslautern. The code for this work was greated by a student group that i was part of. Original GitLab repo can be found [here]([https://gitlab.rhrk.uni-kl.de/algorithm-accountability-lab/fairytale_tokenexplore](https://gitlab.rhrk.uni-kl.de/algorithm-accountability-lab/manim-animations).

The resulting homepage serves as supporting, interactive material for the book ["Weiss die KI, dass sie nichts weiß?"](https://www.penguin.de/buecher/katharina-zweig-weiss-die-ki-dass-sie-nichts-weiss-/paperback/9783453219076) by Katharina Zweig. The resulting homepage can be found [here](https://aalab-weiss-die-ki-dass-sie-nichts-weiss.cs.rptu.de/)

Repo stores code for mathematical simulations / visualizations used on the 
Homepage for the book "Weiß die KI, dass sie nichts weiß" in the section "Wie lernt ChatGPT?". 
The repo can be used to replicate / adapt the used animations.

## Setup

### Install manim python package
```
pip install -r requirements.txt
conda install -c conda-forge manim # for conda env
```

### Install local LaTeX interpreter

For Ubuntu/Debian:
```
sudo apt update
sudo apt install texlive
pdflatex --version
```

For MacOS:
```
brew install --cask mactex
pdflatex --version
```

For Windows:

Follow instructions on https://miktex.org/download

## Run Simulation
Execute from repo root: 
```
manim src/manim.py ClassName
```
- where **manim.py** should be the Python file that stores your Manim Python Class
- and **ClassName** should be the name of that Python Class


## Repo Structure

- **manin.cfg** can be used to configure CLI commands (e.g. change FPS, video quality, media directory)
    - for development set quality to low (480p) to reduce computation requirements
- **media_input** folder stores media that can be used within the Sequences
- **media_output** folder stores the created sequences (videos/run_id/resolution/ClassName.mp4)
- **src** stores the python scripts used to generate Manim Sequences (https://www.manim.community/)
