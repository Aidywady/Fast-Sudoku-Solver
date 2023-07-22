import numpy as np
import sys
import os
import time

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

puzzle_branches = np.array([puzzle, np.zeros(shape=[9,9], dtype=np.int8)], dtype=np.int8)

branch_number = 1

no_of_solutions = 0

check_other_solutions = False

def generate_constraints(puzzle, number): 
    Mask = (puzzle==0) & np.tile(np.all(puzzle != number, axis=1, keepdims=True), [1,9]) & np.tile(np.all(puzzle != number, axis=0, keepdims=True), [9, 1]) & np.tile(np.all(puzzle.reshape(3, 3, 3, 3).swapaxes(1,2).reshape(9, 9) != number, axis=1, keepdims=True), [1,9]).reshape(3, 3, 3, 3).swapaxes(1,2).reshape(9,9)
    return Mask

def generate_candidate_count(puzzle):
    masks_sum = np.zeros([9,9])
    for i in range(9):
        masks_sum += generate_constraints(puzzle, i+1) 
    return masks_sum

def unique(a):
    a_offset = a + np.tile(np.arange(0 , 90, 10).reshape(9, 1), [1, 9])
    unique_repitions = np.bincount(a_offset.ravel(), minlength=90).reshape(9, 10) 
    no_of_unique = (unique_repitions[:, 1:10]!=0).sum(axis=1)
    no_of_unique += unique_repitions[:, 0]

    all_unique = np.all(no_of_unique == 9)
    return all_unique

def valid_sudoku(puzzle):
    rows_valid = unique(puzzle)
    columns_valid = unique(puzzle.T)
    boxes_valid = unique(puzzle.reshape(3, 3, 3, 3).swapaxes(1,2).reshape(9, 9))

    valid = rows_valid and columns_valid and boxes_valid

    return valid

def solve_step(puzzle):

    for i in range(9):
        Mask = generate_constraints(puzzle, i+1)
        puzzle += (i+1) * Mask * np.tile(np.count_nonzero(Mask, axis=1, keepdims=True)==1, [1,9])

    for i in range(9):
        Mask = generate_constraints(puzzle, i+1)
        puzzle += (i+1) * Mask * np.tile(np.count_nonzero(Mask, axis=0, keepdims=True)==1, [9,1])

    for i in range(9):
        Mask = generate_constraints(puzzle, i+1)
        puzzle += (i+1) * Mask * np.tile(np.count_nonzero(Mask.reshape(3, 3, 3, 3).swapaxes(1,2).reshape(9, 9), axis=1, keepdims=True)==1, [1,9]).reshape(3, 3, 3, 3).swapaxes(1,2).reshape(9,9)

    return puzzle

def read_puzzle_branch(puzzle_branches, branch_number):
    branch = abs(puzzle_branches[0, :, :] * (puzzle_branches[1, :, :] <= branch_number))

    guessed_number_mask = (puzzle_branches[0, :, :] < 0) & (puzzle_branches[1, :, :] == branch_number)

    branch = np.where(guessed_number_mask, -branch, branch)

    return branch

def write_puzzle_branch(puzzle, guess_coord, puzzle_branches, branch_number):
    puzzle_branches[0, :, :] *= puzzle_branches[1, :, :] < branch_number
    puzzle_branches[1, :, :] *= puzzle_branches[1, :, :] < branch_number

    puzzle_branches[1, :, :] += branch_number * (puzzle != 0) * (puzzle_branches[1, :, :] == 0) * (puzzle_branches[0, :, :] == 0)

    puzzle_branches[0, :, :] += puzzle * (puzzle_branches[0, :, :] == 0)
    puzzle_branches[0, guess_coord[0], guess_coord[1]] *= -1  

def branch_predict(puzzle, branch_number, puzzle_branches):

    number = 0
    co_ord = (0, 0)

    while number == 0:

        masks_sum = generate_candidate_count(puzzle)

        if branch_number == 0:
            puzzle = np.full(shape=[9, 9], fill_value=-1)
            return [puzzle, branch_number, puzzle_branches]

        invalid_candidate = 0

        temp_branch = read_puzzle_branch(puzzle_branches, branch_number) 
        if np.any(temp_branch < 0):
            i = np.nonzero(temp_branch < 0)
            invalid_candidate = -temp_branch[i[0][0], i[1][0]]

        masks_sum = np.where(masks_sum==0, 9, masks_sum)
        lowest_no_candidates = np.amin(masks_sum)
        i = np.nonzero(masks_sum==lowest_no_candidates)
        co_ord = (i[0][0], i[1][0])

        for i in range(invalid_candidate, 9):
            if number == 0:
                number = (i+1) * generate_constraints(puzzle, i+1)[co_ord]

        if number == 0:
            branch_number -= 1
            puzzle = read_puzzle_branch(puzzle_branches, branch_number)
            puzzle *= puzzle > 0

    puzzle[co_ord] = number
    write_puzzle_branch(puzzle, co_ord, puzzle_branches, branch_number)
    branch_number += 1

    return [puzzle, branch_number, puzzle_branches]

print("Puzzle:")
print(read_puzzle_branch(puzzle_branches, 0))

for i in range(10):
    puzzle = read_puzzle_branch(puzzle_branches, 0)

    puzzle_branches = np.array([puzzle, np.zeros(shape=[9,9], dtype=np.int8)], dtype=np.int8)

    branch_number = 1

    no_of_solutions = 0

    check_other_solutions = False
    while np.all(puzzle>=0) and ((no_of_solutions < 1 and not check_other_solutions) or (no_of_solutions <= 1 and check_other_solutions)):

        puzzle_start = puzzle.copy()

        puzzle = solve_step(puzzle)

        if np.all(puzzle==puzzle_start):

            if not valid_sudoku(puzzle): 
                puzzle = np.where(puzzle == 0, 9, puzzle)

            puzzle, branch_number, puzzle_branches = branch_predict(puzzle, branch_number, puzzle_branches)

        if np.all(puzzle > 0):

            if valid_sudoku(puzzle): 
                no_of_solutions += 1

                if no_of_solutions == 1:
                    solution = puzzle.copy()

time_elapsed = (time.perf_counter() - start_time) * 1000 / 10

print("Solution:")
print(solution)

print("Solved in ", round(time_elapsed, 3), " milliseconds.")

if no_of_solutions > 1:
    print("This isn't a proper Sudoku however, at least one other solution was found.")

if no_of_solutions == 0:
    print("Sorry, this is impossible to solve")

    print("Realised Sudoku can't be solved in", round(time_elapsed, 3), "second(s).")