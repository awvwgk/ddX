# Download and Installation  {#label_download_and_install}

**Build python:**
```
setup.py test
```
**Test python:**
```
./run_ddx.py
```

**Download the ddX source code** at: 
``` markdown
> git clone git@github.com:ACoM-Computational-Mathematics/ddX.git
```
**Download and install** ddX as follows:
``` markdown
> git clone git@github.com:ACoM-Computational-Mathematics/ddX.git
> cd ddX
> mkdir build
> cd build
> cmake .. 
> make
```
Per default, the library is located in /src.

**Build the documentation** as follows (after you have done the above process):
``` markdown
> cd build
> make docs
```
**To see the documentation**
``` markdown
> cd ddX/build/docs/html
> pwd
```
Copy the link shown by pwd and add /index.html in a web browser
#### Hints and hacks
1. For specifying compilers use
``` markdown 
cmake -D CMAKE_CXX_COMPILER=/usr/local/bin/g++-11 CMAKE_C_COMPILER=/usr/local/bin/gcc-11 ..
```
**NOTE**: Replace g++-11 and gcc-11 with the compilers you desire

2. Disactivate OpenMP-support:
``` markdown 
cmake -D OPENMP=OFF ..
```
