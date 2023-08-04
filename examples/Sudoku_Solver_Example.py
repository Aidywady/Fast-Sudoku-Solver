"""
Example code for how to solve a single puzzle.
"""

# import required libraries
import numpy as np # Used for the arrays
import time # Used for timing

import Sudoku_Solver_Functions as sudoku # importing the actual module

# single Sudoku puzzle solver
filename = 'puzzle.txt' # The name of the puzzle file

puzzle = sudoku.read_puzzle(filename) # Here we read the puzzle from the given file and convert it to a numpy array

print("Puzzle:") # Print the puzzle in a user friendly manner using the print_puzzle function
sudoku.print_puzzle(puzzle)

start_time = time.perf_counter() # Get the start time

solution, no_of_solutions = sudoku.solve(puzzle.copy()) # Solve the puzzle using the solve function

time_elapsed = (time.perf_counter() - start_time) * 1000 # Work out the time taken to solve the puzzle

print("Solution:") # Print the solution
sudoku.print_puzzle(solution)

print("Solved in ", round(time_elapsed, 3), " milliseconds.") # Then print the time taken to solve the puzzle

if no_of_solutions == 1: # If the puzzle was valid and a solution was found, write the solution to a text file using puzzle_write.
    sudoku.write_puzzle('solution.txt', solution)

if no_of_solutions == 0: # And if the puzzle wasn't valid and no solution was found, tell the user
    print("Sorry, this is impossible to solve.")
