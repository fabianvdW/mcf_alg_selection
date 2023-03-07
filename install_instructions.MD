Install Cygwin with gcc Do step 1. and 2. in here: https://preshing.com/20141108/how-to-install-the-latest-gcc-on-windows/
Now, uninstall cmake if you have it installed.
We will now install cmake within Cygwin: Open a Terminal at the root folder of cygwin and run
```
C:\cygwin64>setup-x86_64.exe -q -P cmake
```
Then open Cygwin.bat:
```
where cmake
< D:\Programme\Cygwin\bin\cmake.exe
```
`where cmake` should output exactly one path which should look similar to my path(and end with YourCyGwinFolder\bin\cmake.exe)
```
git clone https://gitlab.rhrk.uni-kl.de/vdwarth/fachprakt_mcf_with_ml
cd fachprakt_mcf_with_ml
```
From now on, it is assumed you are in the root folder of the repository ($HOME/fachprakt_mcf_with_ml if you followed above).
Getting the lemon sample project to run:
```
cd lemon
mkdir build
cd build
cmake ..
make
```
If everything worked an executable `./src/lemon-project` should exist. At this point lemon is up and running.

Getting netgen to run:
```
cd netgen
make
```
At this point netgen is up and running if a `netgen` executable exists.

Getting gridgen to run:
```
cd gridgen
gcc gridgen.c -lm -o gridgen
```
At this point gridgen is up and running if a `gridgen` executable exists.

Getting goto to run:
```
cd goto
gcc goto.c -lm -I . -o goto
```
At this point goto is up and running if a `goto` exectuable exists.

Getting cs2 to run:
```
cd cs2
make
```
