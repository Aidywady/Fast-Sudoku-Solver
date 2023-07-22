"""
Example code for how to generate a minimal sudoku puzzle. 
A minimal puzzle is a puzzle in which no clue can be removed, otherwise there will be multiple solutions.
"""

# import required libraries
import numpy as np # Used for the arrays

import Sudoku_Solver_Functions as sudoku # importing the actual module

# Random minimal sudoku generator

solution = sudoku.Generate() # Here we generate a random solution using the generate function.

print("Solution:") 
sudoku.print_puzzle(solution) # Then we print the solution in a user friendly manner using the print_puzzle function

puzzle, no_of_clues = sudoku.generate_minimal_puzzle(solution) # Now we generate a random minimal sudoku puzzle from the solution. The function returns a minimal puzzle, as well as the number of clues in the puzzle.

print("Puzzle:")
sudoku.print_puzzle(puzzle) # We print the puzzle with print_puzzle

print(no_of_clues, "clues are provided.") # Then indicate how many clues (numbers) are provided.

sudoku.write_puzzle('puzzle.txt', puzzle) # We save the puzzle as a text file using the write_puzzle function.
sudoku.write_puzzle('solution.txt', solution) # and the solution as a text file as well.