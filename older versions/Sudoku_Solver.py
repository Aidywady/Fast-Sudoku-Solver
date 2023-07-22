# I must tidy up new puzzle_branches array (observed strange behaviours (i.e. broken))
import numpy as np
import sys
import os
import time

import Sudoku_Solver_APIs as sudoku

#so we can return how long it took to solve
start_time = time.perf_counter()

# This section of code reads the sudoku puzzle.txt file annd converts it into a numpy array
data = ''
path = os.getcwd() + '\sudoku puzzle.txt'
if not os.path.isfile(path):
    print("No sudoku puzzle found at", path)
    sys.exit()
print("opening puzzle file at path", path)
with open(path, 'r') as file:
    data = file.read()
data = " ".join(data.replace("-", "0").replace(" ", "").replace("\n", "")) 
if len(data) != 161:
    print("Text file does not contain a valid Sudoku puzzle")
    sys.exit()
puzzle = np.fromstring(data, sep= ' ', dtype=np.int8)
puzzle = puzzle.reshape(9, 9)

print("Puzzle:")
sudoku.print_puzzle(puzzle)

find_all_solutions = False

Solution, no_of_solutions = sudoku.Solve(puzzle.copy(), find_all_solutions)

time_elapsed = (time.perf_counter() - start_time) * 1000

print("Solution:")
sudoku.print_puzzle(Solution)

print("Solved in ", round(time_elapsed, 3), " milliseconds.")
    
# Let the user know if any other solutions were found (annd the number if we are in the correct mode)
if no_of_solutions > 1:
    print("This isn't a proper Sudoku however, at least one other solution was found.")
    
# In the unlikely event that we find no solutions even though we checked at the beginning, just let the user know
if no_of_solutions == 0:
    print("Sorry, this is impossible to solve")
    # Finally print the amount of time it took to realise it can't be solved
    print("Realised Sudoku can't be solved in", round(time_elapsed, 3), "second(s).")