"""
Example code for how to solve a complete database of puzzles
"""

# import required libraries
import numpy as np # Used for the arrays
import time # Used for timing

import Sudoku_Solver_Functions as sudoku # importing the actual module

#Database solver
filename = 'HardestDatabase.txt' # The filename of the database we are solving

puzzle_database, database_length = sudoku.read_database(filename) # Using the database reading function to obtain a numpy array of all the puzzles as well as the number of puzzles

# declaring some variables for timing
total_time = 0.0
max_time = 0.0 # (to represent a very small max)
min_time = 3600000.0 # an hour (to represent a very large min)

# Declaring variables to keep track of
solutions = 0 # The number of puzzles with a single solution (proper sudoku puzzles)
no_solution = 0 # The number of puzzles with no solution
multiple_solutions = 0 # The number of puzzles with more than one solution

check_other_solutions = False # If we want to check how many solutions each puzzle has, set this to True (the solve function will return that the puzzle will have 0, 1 or 2+ solutions). Note that this slows the solve speed.

# Opening a file to save all the solutions to
with open('database_solutions.txt', 'w') as f:
	f.write("# Solutions to " + filename)

# A for loop that will repeat for each puzzle of the database
for number in range(database_length):
	start_time = time.perf_counter() # Get the start time
	
	puzzle = puzzle_database[number, :, :] # Fetch the specific puzzle we want to solve
	
	print("Puzzle", number+1, "/", database_length, ":", end=' ') # Print the puzzle number being solved, but don't end on a newline.
	
	solution, number_of_solutions = sudoku.Solve(puzzle.copy(), check_other_solutions) # The solve function solves the sudoku. It returns the solution, as well as whether the puzzle has 0, 1 or 2+ (if check_other_solutions = True) solutions. 
	
	time_elapsed = (time.perf_counter() - start_time) * 1000 # We work out the time taken to solve using the difference between the initial and current time.
	
	total_time += time_elapsed
	
	# Check for a new fastest or slowest time
	if time_elapsed > max_time:
		max_time = time_elapsed
		
	if time_elapsed < min_time:
		min_time = time_elapsed
	
	if number_of_solutions == 0: # If no solution was found, print this and indicate it in the solutions file.
		print("No solution found.")
		with open('database_solutions.txt', 'a') as f:
			f.write("No solution found.")
		no_solution += 1

	if number_of_solutions == 1: # If one solution is found, print "solved." and save the solution to the solution database.
		print("Solved.")
		sudoku.append_puzzle('database_solutions.txt', solution)
		solutions += 1
        
	if number_of_solutions == 2: # If more than one solution was found, print that it is "not a proper sudoku" and indicate it in the solutions file.
		print("This is not a proper Sodoku.")
		with open('database_solutions.txt', 'a') as f:
			f.write("Not a Proper puzzle (multiple solutions).")
		multiple_solutions += 1
    	
print("Complete database solved!") # Once we leave for loop, print that the database is solved.

# Print summarised information about the database solutions.
if solutions == database_length:
	print("Each puzzle has 1 solution!")
    
else:
	print("Only", solutions, "puzzles have 1 solution...")
    
if no_solution > 0:
	print(no_solution, "puzzles have no solution.")
    
if multiple_solutions > 0:
	print(multiple_solutions, "puzzles are not proper (have more than one solution)")

# Print some useful information about solve speed, average times, etc. 
print("\n") 
print(round(database_length / (total_time / 1000), 1), "puzzles solved per second.")
print("average time:", round((total_time / database_length), 3), "ms")
print("total time:", round((total_time / 1000), 3), "s")
print("slowest time:", round((max_time), 3), "ms")
print("fastest time:", round((min_time), 3), "ms") 
