# Project 1

**Authors:**


# Usage

## Command Line Usage

Two usage options depending on the current working directory. *Both require python to be on your PATH*.

Either run all commands from inside the Project1 directory.

```bash
../Project1>python data/generator.py
../Project1>python lib/utils.py
```

Or run the bash files from the MathforDS directory. *This option only works if you have a bash shell.*

```bash
../MathforDS>bin/proj1.sh
```

## IDE Usage

The simplest method is to open the `Project1` directory in your IDE. All files should be executable using the built in run function, provided this directory is an endpoint of the PATH used by the IDE. *This does not have to be your system path.*


## Custom Usage

If you need to customize the working directories because they won't run. Generate a local fork and edit the `sys.path.append()` command in the header of the files to accomodate any particular path redirecting that you would like to have.

# Data

Data is generated using the `generator.py` file. This generator file stores the data using `numpy` as a compressed file `data.npz`.

The data is recovered by running the `load_data()` method in the `utils.py` file.

An example of importing this file and utilizing the method is contained in the `main.py` file.