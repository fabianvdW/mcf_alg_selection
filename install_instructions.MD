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
If everything worked, this command should work:
```
./src/lemon-project.exe
< Hello World! This is LEMON library here.
< We have a directed graph with 2 nodes and 1 arc.
```
At this point lemon is up and running.

Getting netgen to run:
```
cd netgen
make
./netgen
```
At this point netgen is up and running.