to run 
* clone the repo
* cd into the repo folder
* `chmod -R +x .` to made script utils executable
* execute the `paper_reproducing.sh` script. 

**NB!** requires `python3` command and `dynet` package installed (e.g. with pip) in python. 
There will appear plots of units activation in the plots folder.

example of working google colaboratory (py3) cells
```
!python3 -m pip install dynet

!git clone https://github.com/oserikov/nn-harmony.git
%cd nn-harmony
!chmod -R +x .

!bash paper_reproducing.sh
```
```
from IPython.display import Image, display

listOfImageNames = !for fn in $(ls plots/*/*); do echo $fn; done;

for imageName in listOfImageNames:
    display(Image(filename=imageName))

```