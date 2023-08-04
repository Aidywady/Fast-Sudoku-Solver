# Fast-Sudoku-Solver
## Description
This project provides a very fast python based sudoku solving function suite making use of NumPy. It is useful for any sudoku relating code in python, and includes solving, generating, validating and reading functions.

It can solve even the hardest Sudoku puzzles in under 1 second on modern hardware, and has a typical solve speed of 5-200ms for most puzzles. In comparison the fastest alternative python code found on Githuh only achieved an average solve speed of about 500ms for the harder puzzles.

## Benchmark Results
Reminder: insert some graphs and benchmark results.

## Dependencies
This code works using Numpy arrays. It also uses os to navigate files. As such the following libraries must be installed for the code to work:
  1. NumPy
  2. OS

## Installation Guide
Installing and using this code is quite simple. Python must be installed, and the NumPy and OS libraries must also be installed. Once this is done, follow these steps to use the code provided in this repository:
  1.  Download the repository .zip file.
  2.  Extract the contents of the download to a temporary folder.
  3.  Create a folder in which you will be writing your python code.
  4.  Copy the *Sudoku_Solver_Functions.py* file from the temporary folder to the folder you created in step 3.
  5.  Create a python script in that folder and open it in your prefered Python code editor/IDE.
  6.  Import the Sudoku_Solver_Functions code at the beginning of the script file using the following import code:
``` python
import Sudoku_Solver_Functions as sudoku
```
  7. Once this has all been done, one can access all the functions provided using:
``` python
sudoku.function_name(...)
```
All the functions and their explanations are explained later in the Readme under **User Reference Guide**.

Furthermore, 3 examples have been provided in the examples folder of the repository. These should run out of the box provided python and the depended libraries are installed.

## User Reference Guide
