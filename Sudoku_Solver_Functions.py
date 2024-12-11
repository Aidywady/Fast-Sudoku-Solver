# -*- coding: utf-8 -*-
"""
Author: Aidan Gebbie
Created: 26 March 2023
License: GPL-3.0 license
Description: A fast python based sudoku solving Function suite that uses NumPy. Useful for any sudoku relating code in python.
User Reference Guide: https://github.com/Aidywady/Fast-Sudoku-Solver/blob/main/README.md
Dependencies: numpy, os
"""

# import required libraries
import numpy as np
from numpy.lib.stride_tricks import as_strided
import os

repeated_box_rearrange = np.arange(0, 81).repeat(9).reshape(9, 3, 3, 3, 3).swapaxes(2, 3).reshape(9, 9, 9)

# A function to read a text file containing a puzzle, and return the puzzle as a 9x9 numpy array
def read_puzzle(filename):
    data = ""
    if not os.path.isfile(filename):
        print("No sudoku puzzle found at", filename)
        return np.zeros(shape=[9, 9], dtype=np.int8)
    print("opening", filename)
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
    
    print("opening sudoku database", filename)
    
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

def generate_candidates(puzzle):
    number_locations = np.full(shape=[9, 9, 9], fill_value=False)
    
    for i in range(9):
        number_locations[i] = (puzzle != (i+1))
    
    row = number_locations.all(axis=2)[..., None]
    col = number_locations.all(axis=1)[..., None].swapaxes(1, 2)
    box = number_locations.reshape(9, 3, 3, 3, 3).swapaxes(2, 3).reshape(9, 9, 9).all(axis=2).ravel()[repeated_box_rearrange]
    
    candidates = row & col & box & (puzzle==0)
    
    return candidates

# A function to eliminate candidates from the candidate array.
# This is used in tandem with other solving functions.
def eliminate_candidates(change, candidates):
    
    #This method has lower overhead but is slower
    for i in np.array(change.nonzero()).tranpose():
        num = change[i[0], i[1]] - 1
            
        if candidates[num, i[0], i[1]]:
            a = i[0] // 3 * 3
            b = i[1] // 3 * 3
            
            candidates[..., i[0], i[1]] = False
            candidates[num, i[0]] = False 
            candidates[num, ..., i[1]] = False
            candidates[num, a:a+3, b:b+3]= False
        else: return candidates, True
            
    return candidates, False

# A function to find and solve for naked singles (when only one candidate is in a specific cell)
def find_singles(puzzle, candidates, candidates_col_view, candidates_box_view, useful_sum):
    #find naked singles
    
    # A error check that every cell has at least one candidate/already solved. 
    
    if np.count_nonzero(useful_sum) + np.count_nonzero(puzzle) != 81:
        return False, True
    
    temp = (useful_sum == 1).nonzero()
    if temp[0].size == 0:
        return False, False
    
    for row, col in zip(*temp):
        single_pos = candidates[..., row, col].nonzero()[0]
        # if a cell suddenly no longer has a single, then there was an error in the guess
        if single_pos.size == 0:
            return False, True
        
        n = single_pos[0]
        
        puzzle[row, col] = n + 1
        
        candidates[n, row] = False 
        candidates_col_view[n, col] = False
        candidates_box_view[n, row // 3, col // 3] = False
    
    return True, False

# A function to find and solve for hidden singles (when only one cell contains a specific candidate)
def find_hidden_singles(puzzle, candidates, candidates_col_view, candidates_box_view, useful_sums):
    # Using a clever optimisation trick where do rows, columns and boxes simultaneously as one array to minimize the numpy overhead.
    single_candidates = useful_sums == 1
    change = False
     
    # faster than .any() ;speed is essential here
    if np.count_nonzero(single_candidates) != 0:
        change = True
        single_candidates = candidates & (single_candidates[0, ..., None] 
                                        | single_candidates[1, ..., None].swapaxes(1,2)
                                        | single_candidates[2].ravel()[repeated_box_rearrange])
                       
        for n, row, col in zip(*single_candidates.nonzero()):
            
            if not candidates[n, row, col]:
                return False, True
            
            puzzle[row, col] = n + 1
            
            candidates[..., row, col] = False
            candidates[n, row] = False 
            candidates_col_view[n, col] = False
            candidates_box_view[n, row // 3, col // 3] = False
    return change, False

# A function to find locked candidates and remove other candidates (candidates locked in place within one region are used to eliminate candidates within another (intersecting) region)
def find_locked_candidates(candidates, candidates_col_view):
    # Using a clever optimisation trick where do horizontals and verticals similitaneously to minimize the numpy overhead.
    temp_mirrors = np.array([candidates, candidates_col_view]).reshape(2, 9, 3, 3, 3, 3)
    merged_mirrors = np.array([temp_mirrors, temp_mirrors.swapaxes(3, 4)]).reshape(4, 9, 3, 3, 3, 3)
    
    tally = merged_mirrors.any(axis=5)
    check = np.add.reduce(tally, axis=4) == 1
    if check.any():
        locked_candidates = (tally & check[..., None]).swapaxes(3, 4)
        locked_candidates_mask = (locked_candidates | (~locked_candidates.any(axis=4))[..., None]).repeat(3, axis=4).reshape(4, 9, 9 ,9)
    
        unmerging_temp = locked_candidates_mask[:2].reshape(2, 9, 3, 3, 3, 3).swapaxes(3, 4).reshape(2, 9, 9, 9)
        new_candidates = candidates & unmerging_temp[0] & locked_candidates_mask[2] & (unmerging_temp[1] & locked_candidates_mask[3]).swapaxes(1, 2)
        
        if (candidates != new_candidates).any():
            candidates &= new_candidates
            return True
        
    return False 

def find_naked_pairs(candidates, useful_sum):
    old_candidates = candidates.copy()
    
    numbered_candidates = candidates * np.arange(1, 10, 1)[..., None, None]
    sumA = np.add.reduce(numbered_candidates, axis=0)
    sumB = useful_sum == 2
    pairs = sumA * sumB * candidates
    
    # Using a clever optimisation trick where do horizontals and verticals similitaneously to minimize the numpy overhead.
    merged_mirrors = np.array([pairs, pairs.swapaxes(1, 2), pairs.reshape(9, 3, 3, 3, 3).swapaxes(2, 3).reshape(9, 9, 9)]).reshape(3, 9, 9, 9)
    
    counts = np.bincount(merged_mirrors.ravel() + np.arange(0, 1458*3, 18).repeat(9), minlength=1458*3).reshape(3, 9, 9, 18) == 2
    
    if (np.add.reduce(counts, axis=3) > 1).any():
        return False, True
    
    temp1 = np.add.reduce(counts * np.arange(0, 18, 1), axis = 3)
    temp1[temp1 == 0] = 18
    
    naked_pairs = (merged_mirrors == temp1[..., None])
    
    final = naked_pairs | ~naked_pairs.any(axis=3)[..., None]
    
    candidates &= final[0] & final[1].swapaxes(1, 2) & final[2].reshape(9, 3, 3, 3, 3).swapaxes(2, 3).reshape(9, 9, 9)
    
    if (candidates != old_candidates).any():
        return True, False
    
    return False, False

def find_hidden_pairs(candidates,candidates_box_view, useful_sums):
    old_candidates = candidates.copy()
    
    merged_mirrors = np.array([candidates, candidates.swapaxes(1, 2), candidates_box_view.reshape(9, 9, 9)]).reshape(3, 9, 9, 9)
    merged_mirrors *= (useful_sums == 2)[..., None]
    
    sum = np.add.reduce(merged_mirrors, axis=1)
    if candidates.any():
        print(merged_mirrors)
        print(sum)
    
    return False, False

# A function for retreiving previous (incorrect) guesses
def read_puzzle_guess(puzzle, candidates, puzzle_guesses, candidate_guesses, guess_data, guess_number):
    # Update candidates based on the guess number
    candidates[...] = (candidate_guesses > guess_number)
    
    # Update puzzle based on the guess number
    puzzle[...] = puzzle_guesses[0] * (puzzle_guesses[1] <= guess_number)
    
    # Retrieve the coordinates and increment the guess count
    row, col = guess_data[0, guess_number], guess_data[1, guess_number]
    guess_data[2, guess_number] += 1
    number = guess_data[guess_data[2, guess_number], guess_number]
    
    return row, col, number

# A function for saving a guess (in case it turns out to be incorrect)
def write_puzzle_guess(puzzle, row, col, weighting, candidates, puzzle_guesses, candidate_guesses, guess_data, guess_number):
    # Save the current puzzle state
    puzzle_guesses[0] = puzzle
    
    # Update puzzle guesses based on the guess number
    puzzle_guesses[1, (puzzle_guesses[1] >= guess_number)] = -1
    puzzle_guesses[1, (puzzle * puzzle_guesses[1] < 0)] = guess_number
    
    # Update candidate guesses
    candidate_guesses[candidate_guesses > guess_number] = guess_number
    candidate_guesses += candidates
    
    guess_data[:2, guess_number] = (row, col)
    if isinstance(weighting, np.ndarray):
        guess_data[2:11, guess_number] = weighting
    
    return

# A function for making and going back on guesses when logical solving isn't enough
def heuristic_guess(puzzle, candidates, candidates_col_view, candidates_box_view, useful_sum, useful_sums, guess_number, puzzle_guesses, candidate_guesses, guess_data, go_back):
    number = -1
    
    while number == -1:
        # Check if we already tried to guess a candidate, if so, try a different candidate in that cell (using the original heuristics)
        if go_back:
            guess_number -= 1
            
            # If there are no more guesses to return on, quit early
            if guess_number == -1:
                return guess_number
            
            row, col, number = read_puzzle_guess(puzzle, candidates, puzzle_guesses, candidate_guesses, guess_data, guess_number)
            weighting = None
        # If we must make a new guess, we must find a cell with the fewest possible candidates,
        # as well as a candidate with the fewest other locations  (to increase chance of correct guess) 
        else:
            weighting = np.array([useful_sum + useful_sums[0, ..., None], 
                                  useful_sum + useful_sums[1, ..., None].swapaxes(1, 2), 
                                  useful_sum + useful_sums[2].ravel()[repeated_box_rearrange]], dtype=np.int8)
            weighting[..., ~candidates] = 19
            
            i = np.argmin(weighting)
            row, col = divmod(i % 81, 9)
            
            weighting = np.argsort(weighting[(i//729), ..., row, col])
            weighting[useful_sum[row, col]:] = -1
            
            number = weighting[0]
            weighting[0] = 2
        
        # If we couldn't find a candidate to put in a cell, the while loop will repeat. It must go back to a previous guess however.
        go_back = True
        
    write_puzzle_guess(puzzle, row, col, weighting, candidates, puzzle_guesses,  candidate_guesses, guess_data, guess_number)
    
    puzzle[row, col] = number + 1
    guess_number += 1
    
    candidates[..., row, col] = False
    candidates[number, row] = False 
    candidates_col_view[number, col] = False
    candidates_box_view[number, row // 3, col // 3] = False

    return guess_number

# An alternative to the heuristic guessing function. This one randomly picks a candidate.
# It is useful for generating puzzles.
def random_guess(puzzle, candidates, candidates_box_view, useful_sums, guess_number, puzzle_guesses, candidate_guesses, guess_data, go_back):
    number = -1
    
    while number == -1:
        if go_back:
            guess_number -= 1
            
            # If there are no more guesses to return on, quit early
            if guess_number == -1:
                return guess_number
            
            co_ord, number = read_puzzle_guess(puzzle, candidates, puzzle_guesses, candidate_guesses, guess_data, guess_number)
            random_sequence = None
            
            
        else:
            i = np.argmin(puzzle)            
            co_ord = (i // 9, i % 9)
            
            random_sequence = np.random.permutation(np.arange(1, 10) * candidates[..., co_ord[0], co_ord[1]])
            random_sequence = random_sequence[random_sequence != 0]
            random_sequence = np.pad(random_sequence, (0, 9-random_sequence.size))
            random_sequence -= 1
            
            number = random_sequence[0]
            random_sequence[0] = 2
        
        if number == -1:
            
            go_back = True
    
    write_puzzle_guess(puzzle, co_ord, random_sequence, candidates, puzzle_guesses,  candidate_guesses, guess_data, guess_number)
    
    puzzle[co_ord] = number+1
    guess_number += 1
    
    candidates[..., co_ord[0], co_ord[1]] = False
    candidates[number, co_ord[0]] = False 
    candidates[number, ..., co_ord[1]] = False
    candidates_box_view[number, co_ord[0] // 3, co_ord[1] // 3] = False

    return guess_number

# A function for checking that no number is repeated in a row. It is used by the valid_sudoku function
def unique(a):
    return (np.bincount(a.ravel() + np.arange(0, 90, 10).repeat(9), minlength=90).reshape(9, 10)[..., 1:] <= 1).all()

# A function for checking that a sudoku is in fact valid (doesn't break the one rule)
def valid_sudoku(puzzle):
    # It uses the unique function while rearanging the puzzle to check for rows, columns and boxes
    if not unique(puzzle): return False
    if not unique(puzzle.transpose()): return False
    if not unique(puzzle.reshape(3, 3, 3, 3).swapaxes(1,2)): return False
    
    return True

# The actual function that solves a puzzle
def solve(puzzle, check_other_solutions=False, solution = np.zeros(shape=[9, 9], dtype=np.int8)):
    # Check that the puzzle doesn't already break the rule
    if not valid_sudoku(puzzle):
        return np.zeros(shape=[9, 9], dtype=np.int8), 0
    
    # Variable declarations:
    puzzle_guesses = np.zeros(shape=[2, 9, 9], dtype=np.int8)
    candidate_guesses = np.zeros(shape=[9, 9, 9], dtype=np.int8)
    guess_data = np.zeros(shape=[11, 81], dtype=np.int8)
    useful_sum = np.zeros(shape=[9, 9], dtype=np.int8)
    useful_sums = np.zeros(shape=[3, 9, 9], dtype=np.int8)
    guess_number = 0
    number_of_solutions = 0
    rolling = 0
    
    # Update the candidates array based on the puzzle
    candidates = generate_candidates(puzzle)
    candidates_col_view = candidates.swapaxes(1, 2)
    candidates_box_view = np.lib.stride_tricks.sliding_window_view(candidates, (1,3,3), writeable=True)[:, ::3, ::3]
    
    # A simple check for mutiple solutions. If multiple solutions are already picked up here,
    # just add one to solutions and set the check to False (i.e. only look for one m(ore) solution)
    if check_other_solutions and len(np.unique(puzzle)) <= 8:
        check_other_solutions = False
        number_of_solutions = 1
    
    # The loop repeats until 2 solutions are found or until it is found that the puzzle can't be solved.
    # Note that the function returns early if a solution is found and the check_other_solutions variable is set to false (see line 444).
    while guess_number >= 0 and number_of_solutions <= 1:
        error = False
        change = True
        
        
        useful_sum = np.add.reduce(candidates, axis=0)
        # find naked singles
        change, error = find_singles(puzzle, candidates, candidates_col_view, candidates_box_view, useful_sum)
        
        # If we couldn't find naked singles, find hidden singles
        if not (change or error):
            useful_sums = np.add.reduce(np.array([candidates,candidates_col_view, candidates_box_view.reshape(9, 9, 9)]), axis=3)
            
            change, error = find_hidden_singles(puzzle, candidates, candidates_col_view,  candidates_box_view, useful_sums)
        
        # If we couldn't find any naked or hidden singles, try find locked candidates
        
        if not (change or error):
            if rolling == 0:
                change = find_locked_candidates(candidates, candidates_col_view)
            rolling = (rolling + 1) % 10
            
        # If none of the above work, check if the puzzle is solved, or make a guess
        if error or not change:
            # Check that the puzzle is valid (the one rule isn't broken)
            if not error and np.count_nonzero(puzzle) == 81 and valid_sudoku(puzzle):
                # If the puzzle is valid and solved...
                number_of_solutions += 1
                
                # This is only useful if we are checking for multiple solutions and we know a solution already
                # if we find a solution that is not the one we know, then the puzzle isn't unique
                if check_other_solutions and (solution != 0).all() and (puzzle != solution).any():
                    return solution, 2
                
                # Only save the first solution (if there are multiple solutions)
                if number_of_solutions == 1:
                    solution = puzzle.copy()
                    
                # If the code mustn't check for other solutions, return early
                if not check_other_solutions:
                    return solution, number_of_solutions
                    
                error = True
            # If a solution is found, or the one rule is broken, we know in advance to reverse the previous guess   
            
            # Make a guess (or go back on one) heuristically.
            guess_number = heuristic_guess(puzzle, candidates, candidates_col_view, candidates_box_view, useful_sum, useful_sums, guess_number, puzzle_guesses, candidate_guesses, guess_data, error)
    
    # Finally return the solution and number of solutions (0, 1 or 2+)
    return solution, number_of_solutions

# A function to generate a complete Sudoku
def generate():
    # Make an empty puzzle array
    puzzle = np.zeros(shape=[9, 9], dtype=np.int8)
    
    # Variable declarations:
    candidates = np.full(shape=[9, 9, 9], fill_value=True)
    candidates_box_view = np.lib.stride_tricks.sliding_window_view(candidates, (1,3,3), writeable=True)[:, ::3, ::3]
    puzzle_guesses = np.array([puzzle, np.zeros(shape=[9,9])], dtype=np.int16)
    candidate_guesses = np.zeros(shape=[9, 9, 9], dtype=np.int8)
    guess_data = np.zeros(shape=[11, 81], dtype=np.int8)
    useful_sums = np.zeros(shape=[4, 9, 9], dtype=np.int8)
    guess_number = 0
    
    # This loop repeats until a completed puzzle is generated
    # (using mostly the same logic as the solve function)
    while (puzzle == 0).any():
        error = False
        change = True
        
        useful_sums[0] = np.add.reduce(candidates, axis=0)
        # find naked singles
        change, error = find_singles(puzzle, candidates, candidates_box_view, useful_sums)
        
        # If we couldn't find naked singles, find hidden singles
        if not (change or error):
            useful_sums[1:4] = np.add.reduce(np.array([candidates,candidates.swapaxes(1,2), candidates_box_view.reshape(9, 9, 9)]), axis=3)
            
            change, error = find_hidden_singles(puzzle, candidates, candidates_box_view, useful_sums)
        
        # If we couldn't find any naked or hidden singles, try find locked candidates
        if not (change or error):
            change = find_locked_candidates(candidates)
            
        if (not change) or error:
            
            # Check that the puzzle is valid before guessing 
            if not valid_sudoku(puzzle):
                error = True
            
            # Make a random guess.
            guess_number = random_guess(puzzle, candidates, candidates_box_view, useful_sums, guess_number, puzzle_guesses, candidate_guesses, guess_data, error)
    
    # Finally return the completed puzzle
    return puzzle

# The actual function that solves a puzzle
def rate_puzzle(puzzle):
    # Variable declarations:
    puzzle_guesses = np.zeros(shape=[2, 9, 9], dtype=np.int8)
    candidate_guesses = np.zeros(shape=[9, 9, 9], dtype=np.int8)
    guess_data = np.zeros(shape=[11, 81], dtype=np.int8)
    useful_sums = np.zeros(shape=[4, 9, 9], dtype=np.int8)
    guess_number = 0
    number_of_solutions = 0
    
    hidden_singles = 0
    singles = 0
    locked_candidates = 0
    naked_pairs = 0
    hidden_pairs = 0
    guesses = 0
    
    # Update the candidates array based on the puzzle
    candidates = generate_candidates(puzzle)
    candidates_box_view = np.lib.stride_tricks.sliding_window_view(candidates, (1,3,3), writeable=True)[:, ::3, ::3]
    
    # The loop repeats until 2 solutions are found or until it is found that the puzzle can't be solved.
    # Note that the function returns early if a solution is found and the check_other_solutions variable is set to false (see line 444).
    while guess_number >= 0 and number_of_solutions <= 1:
        error = False
        change = False
        
        
        useful_sums[0] = np.add.reduce(candidates, axis=0)
        useful_sums[1:4] = np.add.reduce(np.array([candidates,candidates.swapaxes(1,2), candidates_box_view.reshape(9, 9, 9)]), axis=3)
        
        change, error = find_hidden_singles(puzzle, candidates, candidates_box_view, useful_sums)
        hidden_singles += change
        
        # If we couldn't find naked singles, find hidden singles
        if not (change or error):
            # find naked singles
            change, error = find_singles(puzzle, candidates, candidates_box_view, useful_sums)
            singles += change
            
        # If we couldn't find any naked or hidden singles, try find locked candidates
        if not (change or error):
            change = find_locked_candidates(candidates)
            locked_candidates += change
            
        if not (change or error):
            change, error = find_naked_pairs(candidates, useful_sums)
            naked_pairs += change
            
        if not (change or error):
            change, error = find_hidden_pairs(candidates, candidates_box_view, useful_sums)
            hidden_pairs += change
        
        # If none of the above work, check if the puzzle is solved, or make a guess
        if error or not change:
            # Check that the puzzle is valid (the one rule isn't broken)
            if not error and (puzzle != 0).all() and valid_sudoku(puzzle):
                # If the puzzle is valid and solved...
                difficulty = ''
                
                if guesses > 0:
                    difficulty = 'expert'
                elif naked_pairs + hidden_pairs > 0:
                    difficulty = 'hard'
                elif locked_candidates + singles > 0:
                    difficulty = 'medium'
                else:
                    difficulty = 'easy'
                
                return difficulty
            # If a solution is found, or the one rule is broken, we know in advance to reverse the previous guess   
            
            # Make a guess (or go back on one) heuristically.
            guess_number = heuristic_guess(puzzle, candidates, candidates_box_view, useful_sums, guess_number, puzzle_guesses, candidate_guesses, guess_data, error)
            guesses += 1
            
    # Finally return the solution and number of solutions (0, 1 or 2+)
    return 'unsolveable'

# A function to randomly generate a minimal puzzle from a solution (No given can be removed, else there will be multiple solutions)
# This function can be used with the generate function to generate a random puzzle
def generate_minimal_puzzle(solution):
    # It works by randomly generating an order in which given's are removed,
    # And then trying to remove each given while ensuring that there is only one solution
    permutation = np.random.permutation(np.arange(0, 81))
    
    mask = np.full(shape=[9, 9], fill_value=True)
    
    for i in permutation:
        co_ord = (i // 9, i % 9)
        
        mask[co_ord] = False
        
        if solve(solution * mask, True, solution)[1] != 1:
            mask[co_ord] = True
    
    puzzle = solution * mask
    number_of_clues = np.count_nonzero(puzzle)
    
    difficulty = rate_puzzle(puzzle.copy())
    return puzzle, number_of_clues, difficulty