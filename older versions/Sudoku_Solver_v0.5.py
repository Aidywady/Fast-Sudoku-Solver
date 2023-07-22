import numpy as np
import sys
import os
import time

import Sudoku_Solver_APIs_v0_5 as sudoku

start_time = time.perf_counter()

data = ''
path = os.getcwd() + '\expert sudoku 002.txt'
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

for i in range(10):
    Solution, no_of_solutions = sudoku.Solve(puzzle.copy(), find_all_solutions)

time_elapsed = (time.perf_counter() - start_time) * 1000 / 10

print("Solution:")
sudoku.print_puzzle(Solution)

print("Solved in ", round(time_elapsed, 3), " milliseconds.")

if no_of_solutions > 1:
    print("This isn't a proper Sudoku however, at least one other solution was found.")

if no_of_solutions == 0:
    print("Sorry, this is impossible to solve")

    print("Realised Sudoku can't be solved in", round(time_elapsed, 3), "second(s).")