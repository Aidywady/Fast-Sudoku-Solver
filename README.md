# Fast Sudoku Solver
## Description
This project provides a very fast python based sudoku solving function suite making use of NumPy. It is useful for any sudoku relating code in python, and includes solving, generating, validating and reading functions.

It can solve even the hardest Sudoku puzzles in under 500ms on modern hardware, and has a typical solve speed of 1-20ms for most puzzles. In comparison the fastest alternative python code found on GitHub only achieved an average solve speed of about 500ms for the harder puzzles.

## Benchmark Results
A sample of 1000 puzzles from an easy dataset: 0.61ms average

A sample of 1000 puzzles from a medium dataset: 2.55ms average

A sample of 1000 puzzles from a hard dataset: 14.76ms average

Hardest 375 dataset: 26.55ms average

Hardware: i5 1335U + 32GB DDR4 RAM. (Laptop in performance mode) 

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
Below is a list of all the functions the code offers, as well as how to use each function.
Don’t forget to type sudoku. when calling any function, as shown below:
```python 
sudoku.function_name(...)
```
For example, to call the read_puzzle function, type:
``` python
sudoku.read_puzzle(filename)
```
### Functions
This script currently offers 9 main functions (though other internal functions exist):

  •  Reading a puzzle from a file:
``` python
read_puzzle(filename)
```
Accepts 1 argument (*str* filename) and returns 1 argument (NumPy 9x9 array of np.int8). It opens the file at the designated string path (either a filename in the same folder, or the path and filename to a different folder). The puzzle in that file can be 9x9 or flattened to 1x81 and empty cells can have the following placeholders: ‘0’, ‘.’, ‘-’, ‘*’ or ‘?’. It then returns the puzzle found in that file as a 9x9 NumPy array. If no puzzle/file is found, it returns an array of zeros.

  •  Reading each puzzle from a database:
``` python
read_database(filename)
```
Accepts 1 argument (*str* filename) and returns 2 arguments (NumPy nx9x9 array of np.int8, *int* number of puzzles found). It opens the file at the designated string path (either a filename in the same folder, or the path and filename to a different folder). Each puzzle in that database file must start on a new line, be flattened to 1x81, have a  ‘0’ or ‘.’ placeholder for empty cells, and comment lines must start with a ‘#’. It then returns all the puzzle found in that file as a nx9x9 NumPy array where n is the number of puzzles found, and an int of the number of puzzles found. If no puzzle/file is found, it returns a 9x9 array of zeros, length=0. Larger databases may take more time to process, typically 100 000 puzzles are found per second. It also saves the NumPy array as a .npy file in a /cache/ folder for near instantaneous reads on future runs.

  •  Writing a puzzle to a file (not to be confused with appending to file):
``` python
write_puzzle(filename, puzzle)
```
Accepts 2 argument (*str* filename, NumPy 9x9 array of np.int8) and returns nothing. It creates a file at the designated string path (either a filename in the same folder, or the path and filename to a different folder). It then writes the puzzle as 9 rows of 9 numbers (zeros are replaced with dots (.)).

  •  Appending a puzzle to a file (not to be confused with writing to file):
``` python
append_puzzle(filename, puzzle)
```
Accepts 2 arguments (*str* filename, NumPy 9x9 array of np.int8) and returns nothing. It opens a file at the designated string path (either a filename in the same folder, or the path and filename to a different folder), then appends (at the end) the puzzle as a row of 81 numbers (zeros are replaced with dots (.)).

  •  Printing a puzzle to the console:
``` python
print_puzzle(puzzle)
```
Accepts 1 argument (NumPy 9x9 array of np.int8) and returns nothing. It prints the puzzle to the console in a human readable way (with Unicode symbols for borders and spaces to replace zeros).

  •  Checking that a puzzle has no number repeated in a row/column/box:
``` python
valid_sudoku(puzzle)
```
Accepts 1 argument (NumPy 9x9 array of np.int8) and returns 1 argument (bool whether the puzzle is valid). It checks that the provided puzzle obeys the one rule (i.e., no number occurs more than once in a row/column/box). It accepts incomplete puzzles as well. If the puzzle is valid, it returns True, otherwise it returns False. Note that it only checks a puzzle in its current state (i.e. it can't indicate that a puzzle doesn't have a solution)

  •  Solving a puzzle:
``` python
solve(puzzle, check_other_solutions=False)
```
Accepts 1 argument and 1 optional argument (NumPy 9x9 array of np.int8, *bool* check for 2+ solutions which is *False* at default) and returns 2 arguments (a solved puzzle as a NumPy 9x9 array of np.int8 (or an array of zeros if there is no solution), and *int* the number of solutions found (this can be 0 or 1 solution). If *check_other_solutions = True*, the function will try find one other solution, if a second solution is found, the function will return the first solution and number of solutions found will be 2, indicating more than one solution (this slows the solve speed down marginally). Note that depending on the complexity of the puzzle, this function may take more or less time to run, typically between 1ms and 20ms (the slowest recorded time was an extremely hard puzzle that took 230ms to solve on an i5 1335U laptop).

  •  Generating a complete puzzle (i.e. a solution):
``` python
generate()
```
Accepts no arguments and returns 1 argument (NumPy 9x9 array of np.int8). The function generates a random completed (i.e. no zeros) Sudoku puzzle and returns it as a 9x9 NumPy array. It takes roughly 1-10ms to run.

  •  Generate a minimal puzzle from a solution (ie. a rudimentary puzzle generator):
``` python
generate_minimal_puzzle(solution)
```
Accepts 1 argument (NumPy 9x9 array of np.int8) and returns 2 arguments (NumPy 9x9 array of np.int8, *int* number of givens/clues). The function accepts a Sudoku solution and tries to remove numbers in a random order to create a random minimal Sudoku puzzle (i.e., Sudoku puzzle from which no clue can be removed leaving it a proper Sudoku). The minimal puzzle is then returned, as well as how many clues are left. This function typically takes 50ms to run, and usually finds fairly easy puzzles with ±24 clues. Puzzles are ranked with a rudimentry system into 'easy', 'medium', 'hard' or 'expert'. See rate_puzzle for details.

  •  Generate a minimal puzzle from a solution (ie. a rudimentary puzzle generator):
``` python
rate_puzzle(puzzle)
```
Accepts 1 argument (NumPy 9x9 array of np.int8) and returns 1 argument (string difficulty of the puzzle). 
Puzzle rating is done according to:
 * 'easy' : only hidden singles are required to be found to solve the puzzle.
 * 'medium' : naked singles and locked candidates were required to be found to solve the puzzle.
 * 'hard' : hidden and naked pairs were required to solve the puzzle.
 * 'expert' : more advanced methods were required to solve the puzzle.
 * 'unsolveable' : no solution exists.
