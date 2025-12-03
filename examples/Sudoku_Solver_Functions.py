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
import os

repeated_box_rearrange = np.arange(0, 81).repeat(9).reshape(9, 3, 3, 3, 3).swapaxes(2, 3).reshape(9, 9, 9)

def read_puzzle(filename):
    """
    A function to read a text file containing a puzzle, and return the puzzle as a 9x9 numpy array
    
    The file should be a .txt file with 9 rows of 9 numbers. Blank cells can be represented by: - . ? * ' ' 0

    Parameters
    ----------
    filename : filename to read the puzzle from.
        
    Returns
    -------
    puzzle : a 9x9 numpy array representing the puzzle

    """
    
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

def read_database(filename):
    """
    A function to read a text file containing multiple puzzles, and return the puzzles as nx9x9 array.
    A dataset of puzzles must be formated of rows of 81 numbers (flattened sudoku board) with . or 0 as empty cells.

    The function also has a caching feature that saves the numpy array, making it faster to open on subsequent attempts.

    Parameters
    ----------
    filename : filename to read the puzzle from.

    Returns
    -------
    puzzles : nx9x9 array of all the puzzles in the dataset.
    
    length: n, the number of puzzles in the dataset.
    """
    
    if os.path.dirname(filename) == '':
        path = os.getcwd()
    else:
        path = os.path.dirname(filename)
    
    cache_file = path + '\\cache\\' + os.path.splitext(os.path.basename(filename))[0] + '.npy'
    
    if os.path.isfile(cache_file):
        puzzle = np.array(np.load(cache_file))
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

def write_puzzle(filename, puzzle):
    """
    A function to write a single puzzle to a text file in a 9x9 shape.
    
    Parameters
    ----------
    filename: The name of the file to save the puzzle as.
    
    puzzle: the 9x9 array of the sudoku puzzle.
    """
    string = np.array2string(puzzle, separator=' ').replace("[", "").replace("]", "").replace(" ", "").replace("0", ".")
    
    with open(filename, 'w') as f:
        f.write(string)

def append_puzzle(filename, puzzle):
    """
    # A function to append a single puzzle (flattened to 81x1 shape) to the end of a text file. Useful for adding puzzles to a dataset.
    
    Parameters
    ----------
    filename: Name of the file to append the puzzle to.
    
    puzzle: the 9x9 array of the sudoku puzzle.
    """
    string = np.array2string(puzzle, separator=' ').replace("[", "").replace("]", "").replace(" ", "").replace("\n", "").replace("0", ".")
    
    with open(filename, 'a') as f:
        f.write("\n")
        f.write(string)

def print_puzzle(puzzle):
    """
    A function for printing the puzzle in a user friendly manner to the console.
    It uses some unicode symbols to draw the board.

    Parameters
    ----------
    puzzle : The 9x9 array representing the sudoku puzzle.
    """
    
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
    """
    Generate a 9x9x9 boolean array of candidates for the puzzle.
    This array indicates whether it is currently legal for a number to go in a cell.

    Parameters
    ----------
    puzzle : The 9x9 numpy array of the puzzle.

    Returns
    -------
    candidates : The 9x9x9 boolean array for valid candidates.

    """
    number_locations = np.full(shape=[9, 9, 9], fill_value=False)
    
    for i in range(9):
        number_locations[i] = (puzzle != (i+1))
    
    row = number_locations.all(axis=2)[..., None]
    col = number_locations.all(axis=1)[..., None].swapaxes(1, 2)
    box = number_locations.reshape(9, 3, 3, 3, 3).swapaxes(2, 3).reshape(9, 9, 9).all(axis=2).ravel()[repeated_box_rearrange]
    
    candidates = row & col & box & (puzzle==0)
    
    return candidates

# A function to find and solve for naked singles (when only one candidate is in a specific cell)
def find_singles(puzzle, candidates, candidates_col_view, candidates_box_view, useful_sum):
    """
    A function to find the naked singles in a puzzle.

    Parameters
    ----------
    puzzle : The 9x9 numpy array of the puzzle.
    
    candidates : The 9x9x9 boolean numpy array indicating valid candidates for cells.
    
    candidates_col_view : the column view of the candidates array.
    
    candidates_box_view : The box view of the candidates array.
    
    useful_sum : The sum of candidates in each cell.

    Returns
    -------
    change : Boolean value indicating whether any singles were found.
    
    error : a boolean value indicating whether an error in the puzzle has been found.
    """
    
    # A error check that every cell has at least one candidate/already solved. 
    if np.count_nonzero(useful_sum) + np.count_nonzero(puzzle) != 81:
        return False, True
    
    temp = (useful_sum == 1).nonzero()
    if temp[0].size == 0:
        return False, False
    
    for i in range(len(temp[0])):
        row, col = temp[0][i], temp[1][i]
        
        single_pos = candidates[..., row, col].nonzero()[0]
        
        # if a cell suddenly no longer has a single, then there was an error in the guess
        if single_pos.size == 0:
            return False, True
        
        n = single_pos[0]
        
        puzzle[row, col] = n + 1
        
        candidates[n, row] = False 
        candidates[n, ..., col] = False
        candidates_box_view[n, row // 3, col // 3] = False
    
    return True, False

def find_hidden_singles(puzzle, candidates, candidates_col_view, candidates_box_view, useful_sums):
    """
    A function to find and solve for hidden singles (when only one cell contains a specific candidate).
    
    Parameters
    ----------
    puzzle : 9x9 array of the puzzle.
    
    candidates : 9x9x9 boolean array of the candidates for the puzzle.
    
    candidates_col_view : column view of the candidates array.
    
    candidates_box_view : box view of the candidates array.
    
    useful_sums : sum of the rows columns and box candidates.
    
    Returns
    -------
    change : Boolean value indicating whether any singles were found.
    
    error : a boolean value indicating whether an error in the puzzle has been found.
    """
    
    # optimised to solve rows, columns and boxes simultaneously as one array to minimize the numpy overhead.
    single_candidates = useful_sums == 1
    change = False
     
    # faster than .any() ;speed is essential here
    if np.count_nonzero(single_candidates) != 0:
        change = True
        single_candidates = candidates & (single_candidates[0, ..., None] 
                                        | single_candidates[1, ..., None].swapaxes(1,2)
                                        | single_candidates[2].ravel()[repeated_box_rearrange])
        temp = single_candidates.nonzero()
        for i in range(len(temp[0])):
            n, row, col = temp[0][i], temp[1][i], temp[2][i]
            
            if not candidates[n, row, col]:
                return False, True
            
            puzzle[row, col] = n + 1
            
            candidates[..., row, col] = False
            candidates[n, row] = False 
            candidates[n, ..., col] = False
            candidates_box_view[n, row // 3, col // 3] = False
    return change, False

def find_locked_candidates(candidates, candidates_col_view):
    """
    A function to find locked candidates and remove other candidates (candidates locked in place within one region are used to eliminate candidates within another (intersecting) region)

    Parameters
    ----------
    candidates : 9x9x9 boolean array of the candidates for the puzzle.
    
    candidates_col_view : column view of the candidates array.
    
    Returns
    -------
    change : Boolean value indicating whether any singles were found.
    """
    
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
    """
    Function to find naked pairs in the sudoku board.

    Parameters
    ----------
    candidates : 9x9x9 boolean array of the candidates for the puzzle.

    useful_sums : sum of the rows columns and box candidates.
    
    Returns
    -------
    change : Boolean value indicating whether any singles were found.
    
    error : a boolean value indicating whether an error in the puzzle has been found.
    """
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
    """
    Function to find naked pairs in the sudoku board.
    WARNING: INCOMPLETE

    Parameters
    ----------
    candidates : 9x9x9 boolean array of the candidates for the puzzle.

    useful_sums : sum of the rows columns and box candidates.
    
    Returns
    -------
    change : Boolean value indicating whether any singles were found.
    
    error : a boolean value indicating whether an error in the puzzle has been found.
    """
    
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
    """
    Function to retrieve previous puzzle start before incorrect guess.

    Parameters
    ----------
    puzzle : The 9x9 puzzle array.
        
    candidates : the 9x9x9 boolean array of candidates for the puzzle.
    
    puzzle_guesses : 81x9x9 historic puzzle array.
    
    candidate_guesses : 81x9x9x9 historic candidates array.
    
    guess_data : 11x81 array of historic guesses.
    
    guess_number : The current guess number (number between 1 and 81).

    Returns
    -------
    row : The row of the retrieved guess.
    
    col : The column of the retrieved guess.
    
    number : The next number to guess at that cell.
    """
    
    """
    memory efficient
    # Update candidates based on the guess number
    candidates[...] = (candidate_guesses > guess_number)
    
    # Update puzzle based on the guess number
    puzzle[...] = puzzle_guesses[0] * (puzzle_guesses[1] <= guess_number)
    """
    candidates[...] = candidate_guesses[guess_number-1, ...]
    puzzle[...] = puzzle_guesses[guess_number-1, ...]
    
    # Retrieve the coordinates and increment the guess count
    row, col = guess_data[0, guess_number], guess_data[1, guess_number]
    guess_data[2, guess_number] += 1
    number = guess_data[guess_data[2, guess_number], guess_number]
    return row, col, number

# A function for saving a guess (in case it turns out to be incorrect)
def write_puzzle_guess(puzzle, row, col, weighting, candidates, puzzle_guesses, candidate_guesses, guess_data, guess_number):
    """
    A function for saving a guess (in case it turns out to be incorrect).
    
    Parameters
    ----------
    puzzle : the 9x9 puzzle array.
    
    row : the row of the guess being made.
    
    col : the column of the guess being made.
    
    weighting : the order of guess values to make incase of errors.
    
    candidates : the 9x9x9 boolean array of puzzle candidates.
    
    puzzle_guesses : the 81x9x9 historic puzzle array.
    
    candidate_guesses : the 81x9x9x9 historic candidates array.
    """
    
    """
    memory efficient
    Save the current puzzle state
    puzzle_guesses[0] = puzzle
    
    # Update puzzle guesses based on the guess number
    puzzle_guesses[1, (puzzle_guesses[1] >= guess_number)] = -1
    puzzle_guesses[1, (puzzle * puzzle_guesses[1] < 0)] = guess_number
    
    # Update candidate guesses
    candidate_guesses[candidate_guesses > guess_number] = guess_number
    candidate_guesses += candidates
    """
    candidate_guesses[guess_number-1, ...] = candidates
    puzzle_guesses[guess_number-1, ...] = puzzle
    
    guess_data[:2, guess_number] = (row, col)
    if isinstance(weighting, np.ndarray):
        guess_data[2:11, guess_number] = weighting

# A function for making and going back on guesses when logical solving isn't enough
def heuristic_guess(puzzle, candidates, candidates_col_view, candidates_box_view, useful_sum, useful_sums, guess_number, puzzle_guesses, candidate_guesses, guess_data, go_back):
    """
    A function for making and going back on guesses when logical solving isn't sufficient.
    
    Parameters
    ----------
    puzzle: the 9x9 puzzle array.
    
    candidates : 9x9x9 boolean array of the candidates for the puzzle.
    
    candidates_col_view : column view of the candidates array.
    
    candidates_box_view : box view of the candidates array.
    
    useful_sum : sum of the candidates per cell.
    
    useful_sums : sum of the rows columns and box candidates.
    
    guess_number : the current number of guesses made.
    
    puzzle_guesses : the 81x9x9 historic puzzle guesses array.
    
    candidate_guesses : the 81x9x9x9 historic puzzle candidates array.
    
    guess_data : the 11x81 array of guess positions and order.
    
    go_back : flag to go back on a guess due to an error being found.
    
    Returns
    -------
    guess_number : the new number of guesses made.
    """
    
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
            row, col = i//9%9, i%9
            
            weighting = np.argsort(weighting[(i//729), ..., row, col])
            weighting[useful_sum[row, col]:] = -1
            
            number = weighting[0]
            weighting[0] = 2
        
        # If we couldn't find a candidate to put in a cell, the while loop will repeat. It must go back to a previous guess however.
        go_back = True
    
    write_puzzle_guess(puzzle, row, col, weighting, candidates, puzzle_guesses, candidate_guesses, guess_data, guess_number)
    
    puzzle[row, col] = number + 1
    guess_number += 1
    
    candidates[..., row, col] = False
    candidates[number, row] = False 
    candidates_col_view[number, col] = False
    candidates_box_view[number, row // 3, col // 3] = False

    return guess_number

def random_guess(puzzle, candidates, candidates_box_view, useful_sums, guess_number, puzzle_guesses, candidate_guesses, guess_data, go_back):
    """
    An alternative to the heuristic guessing function. This one randomly picks a candidate.
    It is useful for generating puzzles.

    Parameters
    ----------
    puzzle: the 9x9 puzzle array.
    
    candidates : 9x9x9 boolean array of the candidates for the puzzle.
    
    candidates_box_view : box view of the candidates array.
    
    useful_sum : sum of the candidates per cell.
    
    useful_sums : sum of the rows columns and box candidates.
    
    guess_number : the current number of guesses made.
    
    puzzle_guesses : the 81x9x9 historic puzzle guesses array.
    
    candidate_guesses : the 81x9x9x9 historic puzzle candidates array.
    
    guess_data : the 11x81 array of guess positions and order.
    
    go_back : flag to go back on a guess due to an error being found.
    
    Returns
    -------
    guess_number : the new number of guesses made.
    """
    
    number = -1
    
    while number == -1:
        if go_back:
            guess_number -= 1
            
            # If there are no more guesses to return on, quit early
            if guess_number == -1:
                return guess_number
            
            row, col, number = read_puzzle_guess(puzzle, candidates, puzzle_guesses, candidate_guesses, guess_data, guess_number)
            random_sequence = None
            
            
        else:
            i = np.argmin(puzzle)            
            row, col = i // 9, i % 9
            
            random_sequence = np.random.permutation(np.arange(1, 10) * candidates[..., row, col])
            random_sequence = random_sequence[random_sequence != 0]
            random_sequence = np.pad(random_sequence, (0, 9-random_sequence.size))
            random_sequence -= 1
            
            number = random_sequence[0]
            random_sequence[0] = 2
        
        if number == -1:
            
            go_back = True
    
    write_puzzle_guess(puzzle, row, col, random_sequence, candidates, puzzle_guesses,  candidate_guesses, guess_data, guess_number)
    
    puzzle[row, col] = number+1
    guess_number += 1
    
    candidates[..., row, col] = False
    candidates[number, row] = False 
    candidates[number, ..., col] = False
    candidates_box_view[number,row // 3, col // 3] = False

    return guess_number

def unique(a):
    """
    A function for checking that no number is repeated in a row. 
    It is used by the valid_sudoku function.


    Parameters
    ----------
    a : 9x9 array that must have no repeating values per row.

    Returns
    -------
    unique : boolean value, true if there are no repeating values per row.

    """
    
    return (np.bincount(a.ravel() + np.arange(0, 90, 10).repeat(9), minlength=90).reshape(9, 10)[..., 1:] <= 1).all()

def valid_sudoku(puzzle):
    """
    A function for checking that a sudoku is in fact valid (doesn't break the one rule).
                                                            
    Parameters
    ----------
    puzzle : the 9x9 puzzle array
    
    Returns
    -------
    valid : true if the puzzle is valid (no repeating values per row, column or box).

    """
    # It uses the unique function while rearanging the puzzle to check for rows, columns and boxes
    if not unique(puzzle): return False
    if not unique(puzzle.transpose()): return False
    if not unique(puzzle.reshape(3, 3, 3, 3).swapaxes(1,2)): return False
    
    return True

# The actual function that solves a puzzle
def solve(puzzle, check_other_solutions=False):
    """
    Function to solve a sudoku puzzle.

    Parameters
    ----------
    puzzle : a 9x9 numpy array of np.int8 values representing the puzzle.
    
    check_other_solutions : If true, the function will check if the solution is unique.
    
    Returns
    -------
    solution : 9x9 numpy array representing the solution to the puzzle.
    
    number_of_solutions : o if no solution, or 1 if one solution, or 2 if more than one solution.
        Note : 2 solutions indicates that 2 OR MORE solutions exist.

    """
    
    solution = np.zeros(shape=[9, 9], dtype=np.int8)
    
    # Check that the puzzle doesn't already break the rule
    if not valid_sudoku(puzzle):
        return np.zeros(shape=[9, 9], dtype=np.int8), 0
    
    # Variable declarations:
    puzzle_guesses = np.zeros(shape=[81, 9, 9], dtype=np.int8)
    candidate_guesses = np.zeros(shape=[81, 9, 9, 9], dtype=np.int8)
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
        
        #useful_sum = ones.dot(swapped).T
        #useful_sum = np.add.reduce(candidates, axis=0)
        candidates.sum(axis=0, out=useful_sum)
        # find naked singles
        change, error = find_singles(puzzle, candidates, candidates_col_view, candidates_box_view, useful_sum)
        
        # If we couldn't find naked singles, find hidden singles
        if not (change or error):
            np.array([candidates,candidates_col_view, candidates_box_view.reshape(9, 9, 9)]).sum(axis=3,out=useful_sums)
            
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

def generate():
    """
    A function to generate a random, complete Sudoku puzzle.

    Returns
    -------
    puzzle : A 9x9 numpy array with a randomly generated solved Sudoku puzzle.
    """
    
    # Make an empty puzzle array
    puzzle = np.zeros(shape=[9, 9], dtype=np.int8)
    
    # Variable declarations:
    puzzle_guesses = np.zeros(shape=[81, 9, 9], dtype=np.int8)
    candidate_guesses = np.zeros(shape=[81, 9, 9, 9], dtype=np.int8)
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

    # The loop repeats until 2 solutions are found or until it is found that the puzzle can't be solved.
    # Note that the function returns early if a solution is found and the check_other_solutions variable is set to false (see line 444).
    while guess_number >= 0 and number_of_solutions <= 1:
        error = False
        change = True
        
        #useful_sum = ones.dot(swapped).T
        #useful_sum = np.add.reduce(candidates, axis=0)
        candidates.sum(axis=0, out=useful_sum)
        # find naked singles
        change, error = find_singles(puzzle, candidates, candidates_col_view, candidates_box_view, useful_sum)
        
        # If we couldn't find naked singles, find hidden singles
        if not (change or error):
            np.array([candidates,candidates_col_view, candidates_box_view.reshape(9, 9, 9)]).sum(axis=3,out=useful_sums)
            
            change, error = find_hidden_singles(puzzle, candidates, candidates_col_view,  candidates_box_view, useful_sums)
        
        # If we couldn't find any naked or hidden singles, try find locked candidates
        
        if not (change or error):
            if rolling == 0:
                change = find_locked_candidates(candidates, candidates_col_view)
            rolling = (rolling + 1) % 10
        
        if (not change) or error:
            
            # Check that the puzzle is valid before guessing 
            if not valid_sudoku(puzzle):
                error = True
                
            elif (puzzle != 0).all():
                return puzzle
            
            
            # Make a random guess.
            guess_number = random_guess(puzzle, candidates, candidates_box_view, useful_sums, guess_number, puzzle_guesses, candidate_guesses, guess_data, error)
    
    # Finally return the completed puzzle
    return puzzle

def rate_puzzle(puzzle):
    """
    A function the rate a sudoku puzzle.
    
    Puzzle rating is done according to:
        'easy' : only hidden singles are required to be found to solve the puzzle.
        'medium' : naked singles and locked candidates were required to be found to solve the puzzle.
        'hard' : hidden and naked pairs were required to solve the puzzle.
        'expert' : more advanced methods were required to solve the puzzle.
        'unsolveable' : no solution exists.
        
    Parameters
    ----------
    puzzle : the 9x9 numpy array of the puzzle.
    
    Returns
    -------
    difficulty : a string indicating the difficulty of the puzzle 
        ('easy', 'medium', 'hard', 'expert', or 'unsolveable')
        
    """
    
    # Variable declarations:
    puzzle_guesses = np.zeros(shape=[81, 9, 9], dtype=np.int8)
    candidate_guesses = np.zeros(shape=[81, 9, 9, 9], dtype=np.int8)
    guess_data = np.zeros(shape=[11, 81], dtype=np.int8)
    useful_sum = np.zeros(shape=[9, 9], dtype=np.int8)
    useful_sums = np.zeros(shape=[3, 9, 9], dtype=np.int8)
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
    candidates_col_view = candidates.swapaxes(1, 2)
    candidates_box_view = np.lib.stride_tricks.sliding_window_view(candidates, (1,3,3), writeable=True)[:, ::3, ::3]
    
    # The loop repeats until 2 solutions are found or until it is found that the puzzle can't be solved.
    # Note that the function returns early if a solution is found and the check_other_solutions variable is set to false (see line 444).
    while guess_number >= 0 and number_of_solutions <= 1:
        error = False
        change = False
        
        
        useful_sum = np.add.reduce(candidates, axis=0)
        useful_sums = np.add.reduce(np.array([candidates,candidates_col_view, candidates_box_view.reshape(9, 9, 9)]), axis=3)
        
        change, error = find_hidden_singles(puzzle, candidates, candidates_col_view, candidates_box_view, useful_sums)
        hidden_singles += change
        
        # If we couldn't find naked singles, find hidden singles
        if not (change or error):
            # find naked singles
            change, error = find_singles(puzzle, candidates, candidates_col_view, candidates_box_view, useful_sum)
            singles += change
            
        # If we couldn't find any naked or hidden singles, try find locked candidates
        if not (change or error):
            change = find_locked_candidates(candidates, candidates_col_view)
            locked_candidates += change
            
        #if not (change or error):
        #    change, error = find_naked_pairs(candidates, useful_sums)
        #    naked_pairs += change
            
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
            guess_number = heuristic_guess(puzzle, candidates, candidates_col_view, candidates_box_view, useful_sum, useful_sums, guess_number, puzzle_guesses, candidate_guesses, guess_data, error)
            guesses += 1
            
    # Finally return the solution and number of solutions (0, 1 or 2+)
    return 'unsolveable'

def generate_minimal_puzzle(solution):
    """
    A function to randomly generate a minimal puzzle from a solution (No givens can be removed, else there will be multiple solutions)
    This function can be used with the generate function to generate a random puzzle
    
    Parameters
    ----------
    solution : the 9x9 solution array that the puzzle must be made from.
    
    Returns
    -------
    puzzle : the 9x9 array for minimal puzzle generated from the solution.
    
    number_of_clues : number of values left in the puzzle.
    
    difficulty : the difficulty of the puzzle (see rate_puzzle for details):
        ('easy', 'medium', 'hard', or 'expert')
    """
    
    # It works by randomly generating an order in which given's are removed,
    # And then trying to remove each given while ensuring that there is only one solution
    permutation = np.random.permutation(np.arange(0, 81))
    
    mask = np.full(shape=[9, 9], fill_value=True)
    
    for i in permutation:
        co_ord = (i // 9, i % 9)
        
        mask[co_ord] = False
        
        if solve(solution * mask, True)[1] != 1:
            mask[co_ord] = True
    
    puzzle = solution * mask
    number_of_clues = np.count_nonzero(puzzle)
    
    difficulty = rate_puzzle(puzzle.copy())
    return puzzle, number_of_clues, difficulty
