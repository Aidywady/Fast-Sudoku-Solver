# -*- coding: utf-8 -*-
"""
Author: Aidan Gebbie
Created: 5 April 2023
License: GPL-3.0 license
Description: A fast python based sudoku solving Function suite that uses NumPy. Useful for any sudoku relating code in python.
User Reference Guide: https://github.com/Aidywady/Fast-Sudoku-Solver/blob/main/README.md
Dependencies: numpy, os
"""

# import required libraries
import numpy as np
import os

# A function to read a text file containing a puzzle, and return the puzzle as a 9x9 numpy array
def read_puzzle(filename):
	data = ""
	if not os.path.isfile(filename):
		print("No sudoku puzzle found at", filename)
		return np.zeros(shape=[9, 9], dtype=np.int8)
	print("opening puzzle file at path", filename)
	with open(filename, 'r') as file:
		data = file.read()
	data = " ".join(data.replace("-", "0").replace(".", "0").replace("?", "0").replace("*", "0").replace(" ", "").replace("\n", ""))
	if len(data) != 161:
		print("No sudoku puzzle found at", filename)
		return np.zeros(shape=[9, 9], dtype=np.int8)
	puzzle = np.fromstring(data, sep= " ", dtype=np.int8)
	puzzle = puzzle.reshape(9, 9)
	return puzzle

# A function to read a text file containing multiple puzzles, and return the puzzles as nx9x9 array 
def read_database(filename):
	# The function also has a caching feature that saves the numpy array, making it faster to open on subsequent attempts.
	if os.path.dirname(filename) == '':
		path = os.getcwd()
	else:
		path = os.path.dirname(filename)
	
	cache_file = path + '\\cache\\' + os.path.splitext(os.path.basename(filename))[0] + '.npy'
	
	if os.path.isfile(cache_file):
		puzzle = np.load(cache_file)
		return puzzle.copy(), np.shape(puzzle)[0]
		
	data = ''
	length = 0
	
	puzzle = np.zeros(shape=0, dtype=np.int8)
	if not os.path.isfile(filename):
		print("No sudoku database found at", filename)
		return np.zeros(shape=[9, 9], dtype=np.int8), 0
	
	print("opening sudoku database at path", filename)
	
	for line in open(filename, 'r'):
		if not line.strip().startswith("#"):
			data += line[0:81]
			length += 1
			
			if length % 10000 == 0:
				data = " ".join(data.replace(".", "0"))
				puzzle = np.append(puzzle, np.fromstring(data, sep= " ", dtype=np.int8))
				data = ''
				if length % 100000 == 0:
					print(length, " puzzles found...")
    
	if length == 0:
		print("No sudoku puzzle found at", filename)
		return np.zeros(shape=[9, 9], dtype=np.int8), 0
					
	data = " ".join(data.replace(".", "0"))
	
	puzzle = np.append(puzzle, np.fromstring(data, sep= " ", dtype=np.int8)).reshape(length, 9, 9)
	
	if not os.path.exists(os.path.dirname(cache_file)):
		os.makedirs(os.path.dirname(cache_file))
	
	np.save(cache_file, puzzle)
	
	return puzzle.copy(), length

# A function to write a single puzzle to a text file in a 9x9 shape.
def write_puzzle(filename, puzzle):
	string = np.array2string(puzzle, separator=' ').replace("[", "").replace("]", "").replace(" ", "").replace("0", ".")
	with open(filename, 'w') as f:
		f.write(string)

# A function to append a singe puzzle to the end of a text file in a 81x1 shape. Useful for adding puzzles to a dataset
def append_puzzle(filename, puzzle):
	string = np.array2string(puzzle, separator=' ').replace("[", "").replace("]", "").replace(" ", "").replace("\n", "").replace("0", ".")
	with open(filename, 'a') as f:
		f.write("\n")
		f.write(string)

# A function for printing the puzzle and solution in a user friendly manner (using some unicode symbols)
def print_puzzle(puzzle):
	print("╔═══════╤═══════╤═══════╗")
	for row in range(9):
		if row % 3 == 0 and row != 0:
			print("╟───────┼───────┼───────╢")
		for col in range(9):
			if col == 0 :
				print("║ ", end="")
			
			if col % 3 == 0 and col != 0:
				print("│ ", end="")
				
			print(str(puzzle[row, col]).replace("0", " "), end=" ")
			
			if col == 8:
				print("║")
				
	print("╚═══════╧═══════╧═══════╝")

# A pair of internally used functions that can be faster than np.any and np.all on smaller arrays.
def myany(array):
	a = array.ravel()
	for i in range(len(a)):
		if a[i]: return True
	return False

def myall(array):
	a = array.ravel()
	for i in range(len(a)):
		if not a[i]: return False
	return True

# A function to eliminate candidates from the candidate array.
# This is used in tandem with other solving functions.
def eliminate_candidates(change, candidates):
	list = np.array(np.nonzero(change))
	
	#This method has lower overhead but is slower
	if len(list[0, :]) <= 15:
		for i in range(len(list[0, :])):	
			num = change[list[0, i], list[1, i]] - 1
			
			if candidates[num, list[0, i], list[1, i]] == False:
				return candidates, True
			
			a = int(list[0, i] / 3) * 3
			b = int(list[1, i] / 3) * 3
			
			candidates[:, list[0, i], list[1, i]] = False
			candidates[num, list[0, i], :] = False 
			candidates[num, :, list[1, i]] = False
			candidates[num, a:a+3, b:b+3]= False
			
		return candidates, False
	
	#else this method has higher overhead but is faster
	list = np.unique(change)[1:]
	length = len(list)
		
	number_locations = np.full(shape=[length , 9, 9], fill_value=False)
		
	for i in range(length):
		number_locations[i, :, :] = (change != list[i])
		
	row = np.all(number_locations, axis=2, keepdims=True)
	col = np.all(number_locations, axis=1)
	box = np.all(number_locations.reshape(length, 3, 3, 3, 3).swapaxes(2,3).reshape(length, 9, 9), axis=2, keepdims=True).repeat(9).reshape(length, 3, 3, 3, 3).swapaxes(2,3).reshape(length, 9, 9)
	
	for i in range(length):
		
		candidates[list[i]-1, :, :] *= row[i, :] * col[i, :] * box[i, :]
	
	candidates *= (change == 0)
		
	return candidates, False

# A function to find and solve for naked singles (when only one candidate is in a specific cell)
def find_singles(puzzle, candidates):
	#find naked singles
	temp = np.sum(candidates, axis = 0)
	
	# A error check that every cell has at least one candidate/already solved.
	if myany(temp + puzzle == 0):
		return puzzle, candidates, True, True
	
	single_candidates = (temp == 1)
	
	change = False
	error = False
	
	if myany(single_candidates):
		change = True
		
		temp = single_candidates * np.sum(candidates * np.arange(1, 10).repeat(81).reshape(9, 9, 9), axis=0)
		
		puzzle += temp
		candidates, error = eliminate_candidates(temp, candidates)
	
	return puzzle, candidates, change, error

# A function to find and solve for hidden singles (when only one cell contains a specific candidate)
def find_hidden_singles(puzzle, candidates):
	# Using a clever optimisation trick where a do rows, columns and boxes simultaneously as one array to minimize the numpy overhead.
	merged_mirrors = np.concatenate([candidates, candidates.swapaxes(1, 2), candidates.reshape(9, 3, 3, 3, 3).swapaxes(2,3).reshape(9, 9, 9)]).reshape(3, 9, 9, 9)
	
	counts = np.sum(merged_mirrors, axis=3) == 1
	
	list = np.any(counts[0] + counts[1] + counts[2], axis=1)
	
	change = False
	error = False
	
	temp = np.zeros_like(puzzle)
	
	for i in range(9):
		if list[i]:
			temp += (i+1) * (temp + puzzle == 0) * candidates[i, :, :] * (counts[0, i].reshape(9, 1) + counts[1, i] + counts[2, i, :].repeat(9).reshape(3, 3, 3, 3).swapaxes(1,2).reshape(9, 9))
			change = True
			
	if change:
		puzzle += temp
		candidates, error = eliminate_candidates(temp, candidates)
	
	return puzzle, candidates, change, error

# A function to find locked candidates and remove other candidates (candidates locked in place within one region are used to eliminate candidates within another (intersecting) region)
def find_locked_candidates(candidates):
	old_candidates = candidates.copy()
	
	temp_mirrors = np.concatenate((candidates, candidates.swapaxes(1, 2))).reshape(2, 9, 3, 3, 3, 3)
	
	# Using a clever optimisation trick where do horizontals and verticals similitaneously to minimize the numpy overhead.
	merged_mirrors = np.concatenate((temp_mirrors, temp_mirrors.swapaxes(3, 4)))
	
	tally = np.sum(merged_mirrors.reshape(4, 9, 27, 3), axis=3).reshape(4, 9, 9, 3) != 0
	check = (np.sum(tally, axis=3) == 1)
	if np.any(check):
		temp = (tally.ravel() * check.repeat(3)).reshape(4, 9, 3, 3, 3).swapaxes(3, 4)
		
		temp_final = (temp.ravel() | (np.sum(temp, axis=4) == 0).repeat(3)).repeat(3).reshape(4, 9, 9 ,9)
	
		unmerging_temp = temp_final[0:2].reshape(2, 9, 3, 3, 3, 3).swapaxes(3,4).reshape(2, 9, 9, 9)
	
		candidates *= unmerging_temp[0] * temp_final[2] * (unmerging_temp[1] * temp_final[3]).swapaxes(1, 2)
	
		if np.any(candidates != old_candidates):
			return candidates, True
		
	return candidates, False

# A function for retreiving previous (incorrect) guesses
def read_puzzle_guess(puzzle_guesses,  candidate_guesses, guess_number):
	branch = abs(puzzle_guesses[0, :, :] * (puzzle_guesses[1, :, :] <= guess_number))
	
	guessed_number_mask = (puzzle_guesses[0, :, :] < 0) * (puzzle_guesses[1, :, :] == guess_number)
	branch = np.where(guessed_number_mask, -branch, branch)
	
	branch_candidates = ( candidate_guesses >= guess_number)
	return branch, branch_candidates

# A function for saving a guess (in case it turns out to be incorrect)
def write_puzzle_guess(puzzle, co_ord, number, candidates, puzzle_guesses,  candidate_guesses, guess_number):
	puzzle_guesses *= puzzle_guesses[1, :, :] < guess_number
	
	puzzle_guesses[0, :, :] += puzzle * (puzzle_guesses[0, :, :] == 0)
	puzzle_guesses[0, co_ord[0], co_ord[1]] *= -1
	
	puzzle_guesses[1, :, :] += guess_number * (puzzle != 0) * (puzzle_guesses[0, :, :] * puzzle_guesses[1, :, :] == 0)
	
	candidate_guesses = np.where( candidate_guesses >= guess_number, guess_number - 1,  candidate_guesses)
	
	candidate_guesses = np.where(candidates, guess_number,  candidate_guesses)
	candidate_guesses[number-1, co_ord[0], co_ord[1]] = guess_number-1
	
	return  candidate_guesses

# A function for making and going back on guesses when logical solving isn't enough
def heuristic_guess(puzzle, guess_number, puzzle_guesses,  candidate_guesses, go_back, candidates):
	number = 0
	co_ord = (0, 0)
	
	while number == 0:
		# Check if we already tried to guess a candidate, if so, try a different candidate in that cell
		if go_back:
			guess_number -= 1
			
			# If there are no more guesses to return on, quit early
			if guess_number == 0:
				return [puzzle, guess_number, puzzle_guesses, candidates,  candidate_guesses]
			
			puzzle, candidates = read_puzzle_guess(puzzle_guesses,  candidate_guesses, guess_number)
			
			i = np.argmin(puzzle)
			
			co_ord = (int(i/9), i % 9)
			
			# If there is only one other choice, choose it.
			if np.sum(candidates[:, co_ord[0], co_ord[1]]) <= 1:
				number = np.sum(candidates[:, co_ord[0], co_ord[1]] * np.arange(1, 10, 1))
			
			# Else, find the candidate that has the fewest other locations (to increase chance of correct guess)
			else:
				temp1 = np.sum(candidates[:, co_ord[0], :], axis=1)
				temp2 = np.sum(candidates[:, :, co_ord[1]], axis=1)
				a = int(co_ord[0] / 3) * 3
				b = int(co_ord[1] / 3) * 3
				temp3 = np.sum(candidates[:, a:a+3, b:b+3].reshape(9, 9), axis=1)
				
				temp4 = np.array([temp1, temp2, temp3]) * candidates[:, co_ord[0], co_ord[1]]
				
				temp4[temp4 == 0] = 10
				
				weighting = np.min(temp4, axis=0)
				
				number = np.argmin(weighting) + 1
		
		# If we must make a new guess, we must find a cell with the fewest possible candidates,
		# as well as a candidate with the fewest other locations  (to increase chance of correct guess) 
		else:
			weighting = np.zeros(shape=[9, 9, 9], dtype=np.int8)
			
			temp1 = np.sum(candidates, axis=1).repeat(9).reshape(9, 9, 9).swapaxes(1, 2)
			temp2 = np.sum(candidates, axis=2).repeat(9).reshape(9, 9, 9)
			temp3 = np.sum(candidates.reshape(9, 3, 3, 3, 3).swapaxes(2,3).reshape(9, 9, 9), axis=2).repeat(9).reshape(9, 3, 3, 3, 3).swapaxes(2,3).reshape(9, 9, 9)
			
			weighting = np.min([temp1, temp2, temp3], axis=0)
			
			masks_sum = np.sum(candidates, axis=0)
			
			combined = masks_sum + weighting
			
			combined[candidates==False] = 20
			
			i = np.argmin(combined)
			
			co_ord = (int((i % 81)/9), i % 9)
			
			number = int(i/81) + 1
			
			if candidates[number-1, co_ord[0], co_ord[1]] == False:
				number = 0
		
		# If we couldn't find a candidate to put in a cell, we must revert to a previous guess
		if number == 0:
			go_back = True
	
	puzzle[co_ord] = number
	
	candidate_guesses = write_puzzle_guess(puzzle, co_ord, number, candidates, puzzle_guesses,  candidate_guesses, guess_number)
	
	guess_number += 1
	
	temp = np.zeros(shape=[9, 9], dtype=np.int8)
	temp[co_ord] = number
	candidates, error = eliminate_candidates(temp, candidates)

	return [puzzle, guess_number, puzzle_guesses, candidates,  candidate_guesses]

# An alternative to the heuristic guessing function. This one randomly picks a candidate.
# It is useful for generating puzzles.
def random_guess(puzzle, guess_number, puzzle_guesses,  candidate_guesses, go_back, candidates):
	number = 0
	co_ord = (0, 0)
	
	while number == 0:
		temp = 0
		
		if go_back:
			guess_number -= 1   
			puzzle, candidates = read_puzzle_guess(puzzle_guesses,  candidate_guesses, guess_number)
						
		temp = np.argmin(puzzle)			
		co_ord = (int(temp/9), temp % 9)
		random_sequence = np.random.permutation(np.arange(1, 10, 1) * candidates[:, co_ord[0], co_ord[1]])
		
		for i in range(0, 9):
			if number == 0:
				number = random_sequence[i]
		
		if number == 0:
			go_back = True
			
	puzzle[co_ord] = number
	
	candidate_guesses = write_puzzle_guess(puzzle, co_ord, number, candidates, puzzle_guesses,  candidate_guesses, guess_number)
	
	guess_number += 1
	
	temp = np.zeros(shape=[9, 9], dtype=np.int8)
	temp[co_ord] = number
	candidates, error = eliminate_candidates(temp, candidates)

	return [puzzle, guess_number, puzzle_guesses, candidates,  candidate_guesses]

# A function for checking that no number is repeated in a row. It is used by the valid_sudoku function
def unique(a):
	return myall(np.bincount(a.ravel() + np.arange(0, 90, 10).repeat(9), minlength=90).reshape(9, 10)[:, 1:10] <= 1)

# A function for checking that a sudoku is in fact valid (doesn't break the one rule)
def valid_sudoku(puzzle):
	# It uses the unique function while rearanging the puzzle to check for rows, columns and boxes
	if not unique(puzzle): return False
	if not unique(puzzle.swapaxes(0,1)): return False
	if not unique(puzzle.reshape(3, 3, 3, 3).swapaxes(1,2)): return False
	
	return True

# The actual function that solves a puzzle
def solve(puzzle, check_other_solutions=False):
	# Check that the puzzle doesn't already break the rule
	if not valid_sudoku(puzzle):
		return puzzle, 0
	
	if len(np.unique(puzzle)) <= 8:
		return puzzle, 2
	
	# Variable declarations:
	solution = np.zeros(shape=[9, 9], dtype=np.int8)
	candidates = np.full(shape=[9, 9, 9], fill_value=True)
	puzzle_guesses = np.array([puzzle, np.zeros(shape=[9,9])], dtype=np.int8)
	candidate_guesses = np.zeros(shape=[9, 9, 9], dtype=np.int8)
	guess_number = 1
	number_of_solutions = 0
	
	# Update the candidates array based on the puzzle
	candidates, error = eliminate_candidates(puzzle, candidates)
	
	# The loop repeats until 2 solutions are found or until it is found that the puzzle can't be solved.
	# Note that the function returns early if a solution is found and the check_other_solutions variable is set to false (see line 444).
	while guess_number > 0 and number_of_solutions <= 1:
		error = False
		change = False
		# find naked singles
		puzzle, candidates, change, error = find_singles(puzzle, candidates)
		
		# If we couldn't find naked singles, find hidden singles
		if not (change or error):
			puzzle, candidates, change, error = find_hidden_singles(puzzle, candidates)
		
		# If we couldn't find any naked or hidden singles, try find locked candidates
		if not (change or error):
			candidates, change = find_locked_candidates(candidates)
		
		# If none of the above work, check if the puzzle is solved, or make a guess
		if (not change) or error:
			# Check that the puzzle is valid (the one rule isn't broken)
			if not error and myall(puzzle > 0) and valid_sudoku(puzzle):
				# If the puzzle is valid and solved...
				number_of_solutions += 1
				# Only save the first solution (if there are multiple solutions)
				if number_of_solutions == 1:
					solution = puzzle.copy()
					
				# If the code mustn't check for other solutions, return early
				if not check_other_solutions:
					return solution, number_of_solutions
					
				error = True
			# If a solution is found, or the one rule is broken, we know in advance to reverse the previous guess   
			
			# Make a guess (or go back on one) heuristically.
			puzzle, guess_number, puzzle_guesses, candidates,  candidate_guesses = heuristic_guess(puzzle, guess_number, puzzle_guesses,  candidate_guesses, error, candidates)
	
	# Finally return the solution and number of solutions (0, 1 or 2+)
	return solution, number_of_solutions

# A function to generate a complete Sudoku
def generate():
	# Make an empty puzzle array
	puzzle = np.zeros(shape=[9, 9], dtype=np.int8)
	
	# Variable declarations:
	candidates = np.full(shape=[9, 9, 9], fill_value=True)
	puzzle_guesses = np.array([puzzle, np.zeros(shape=[9,9])], dtype=np.int8)
	candidate_guesses = np.zeros(shape=[9, 9, 9], dtype=np.int8)
	guess_number = 1
	
	# Update the candidates array based on the puzzle
	candidates, error = eliminate_candidates(puzzle, candidates)
	
	# This loop repeats until a completed puzzle is generated
	# (using mostly the same logic as the solve function)
	while myany(puzzle == 0):
		error = False
		change = False
		puzzle, candidates, change, error = find_singles(puzzle, candidates)
		
		if not (change or error):
			puzzle, candidates, change, error = find_hidden_singles(puzzle, candidates)
		
		if not (change or error):
			candidates, change = find_locked_candidates(candidates)
		  
		if (not change) or error:
			
			# Check that the puzzle is valid before guessing 
			if not valid_sudoku(puzzle):
				error = True
			
			# Make a random guess.
			puzzle, guess_number, puzzle_guesses, candidates,  candidate_guesses = random_guess(puzzle, guess_number, puzzle_guesses,  candidate_guesses, error, candidates)
	
	# Finally return the completed puzzle
	return puzzle

# A function to randomly generate a minimal puzzle from a solution (No given can be removed, else there will be multiple solutions)
# This function can be used with the generate function to generate a random puzzle
def generate_minimal_puzzle(solution):
	# It works by randomly generating an order in which given's are removed,
	# And then trying to remove each given while ensuring that there is only one solution
	permutation = np.random.permutation(np.arange(0, 81, 1))
	
	mask = np.full(shape=[9, 9], fill_value=True)
	
	number_of_solutions = 0
	
	for i in permutation:
		co_ord = (int(i/9), i % 9)
		
		mask[co_ord] = False
		
			
		temp, number_of_solutions = solve(solution * mask, True)
			
		if number_of_solutions != 1:
			mask[co_ord] = True
	
	puzzle = solution * mask
	
	temp, number_of_solutions = solve(puzzle.copy(), True)
	number_of_clues = np.count_nonzero(puzzle)
	
	return puzzle, number_of_clues
