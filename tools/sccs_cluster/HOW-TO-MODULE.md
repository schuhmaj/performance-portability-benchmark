# How to Add Local Modules

## Overview

[Modules](https://modules.readthedocs.io/en/latest/)
is a package that is often utilized in the HPC
context to allow users to effortlessly adapt
their environment by, e.g., switching
to a different compiler.

Usually, you don't have the privileges to install
any new software system-wide.
However, sometimes you require a library multiple
times. Then, it might be suitable to
install it in your local space. To faciliate build
tools like `CMake` the finding of the library
without the need to specify `CMAKE_PREFIX_PATH` each
time, we can append our local installation
directory to the module path.

## Step by Step

Example using [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp).

### Create a local installation directory

```bash
cd ~
mkdir modules
mkdir modules-files
```

and append the `modules-files` directory
to your module path by adding to you
`.bashrc` or `.zshrc` file:
```bash
export MODULEPATH=$MODULEPATH:~/modules-files
```
### Build and Install your Software

```bash
git clone git@github.com:AdaptiveCpp/AdaptiveCpp.git
git checkout v24.06.0
cd AdaptiveCpp
# Load required modules, e.g. module load LLVM
mkdir build && cd build
cmake .. -G Ninja -... # append options
cmake --build .
cmake --install . --prefix ~/modules/AdaptiveCpp/24.06.0/
```

### Create a module file

```bash
cd ~/modules-files
mkdir AdaptiveCpp && cd AdaptiveCpp
touch 24.06.0
```

Modify the content of the text file `24.06.0`
to the content below:

```text
#%Module1.0
#
# AdaptiveCpp Module
#
proc ModulesHelp { } {
   puts stderr "This module loads the environment for AdaptiveCpp 24.06.0"
}

module-whatis "Loads AdaptiveCpp 24.06.0"

# Set the root directory for AdaptiveCpp
set root ~/modules/AdaptiveCpp/24.06.0

# Prepend the AdaptiveCpp bin directory to the PATH environment variable
prepend-path PATH $root/bin

# Prepend the AdaptiveCpp lib directory to the LD_LIBRARY_PATH environment variable
prepend-path LD_LIBRARY_PATH $root/lib

# Prepend the AdaptiveCpp man directory to the MANPATH environment variable
prepend-path MANPATH $root/share/man
```

### Congratulation

You now have an activate-able and deactivate-able
[AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp)
installation.
Activate by

```bash
module load AdaptiveCpp/24.06.0
```

Deactivate by

```bash
module unload AdaptiveCpp/24.06.0
```