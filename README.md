# MLR_BP2_Repo
getting communication between ue4 and external python script working

## How to use

1) open ue4.24 make a new third person shooter game c++, name it MLR_BP2. Close the game and VS
2) copy the plugins folder into the project directory
3) go into source and get the build.cs file. Copy that into the //project directory/source to overwrite
4) open up the ue4 editor, click on the character, add new component. Show all classes. Search machine learning. Add new machine learning base component named bp2_mlb
5) Add new machine learning remote component named bp2_mlr
6) Compile and save. Close the game and visual studio again
7) Copy the entirety of the source folder, content, and config into the project directory to overwrite
8) Open the game and construct the blueprint based on the image bp.png here
9) In the blueprint, be sure to configure the machine learning remote component to use hello.py
10) start up the server (StartupServer.bat or server.py)
11) Play the game.
