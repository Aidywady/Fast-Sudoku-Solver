import numpy as np
import sys
import os
import time

start_time = time.perf_counter()

data = ''
path = os.getcwd() + '\hard sudoku 001.txt'
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

last_puzzle = np.zeros([9, 9], dtype=np.int8)

candidates = np.full(shape=[9, 9, 9], fill_value=True, dtype=bool)

puzzle_branches = np.array([puzzle, np.zeros(shape=[9,9])], dtype=np.int8)

branch_number = 1

no_of_solutions = 0

check_other_solutions = False

still_solveable = True

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

def coord_constraint(puzzle, co_ord, number):
    if puzzle[co_ord] > 0: return False
    if myany(puzzle[co_ord[0], :] == number): return False
    if myany(puzzle[:, co_ord[1]] == number): return False
    if myany(puzzle.reshape(3, 3, 3, 3).swapaxes(1,2).reshape(9, 9)[(int(co_ord[0] / 3)*3 +int(co_ord[1]/3)), :] == number): return False
    return True

def eliminate_candidates(puzzle, last_puzzle, candidates):
    candidates *= (puzzle == 0)

    change = puzzle * (last_puzzle != puzzle)

    list = np.unique(change)  

    for i in range(len(list)-1):
        number_locations = (change != list[i+1])

        candidates[list[i+1]-1, :, :] *= np.all(number_locations, axis=1, keepdims=True) * np.all(number_locations, axis=0, keepdims=True) * ((np.full(shape=[9, 9], fill_value=True) * np.all(number_locations.reshape(3, 3, 3, 3).swapaxes(1,2).reshape(9, 9), axis=1, keepdims=True)).reshape(3, 3, 3, 3).swapaxes(1,2).reshape(9,9))

    return candidates

def logical_solve(puzzle, candidates):
    change = False

    for i in range(9):
        if myany(candidates[i, :, :]):
            certaintyr = np.sum(candidates[i, :, :], 1)==1
            certaintyc = np.sum(candidates[i, :, :], axis=0)==1  

            if myany(certaintyr + certaintyc):
                puzzle += (i+1) * (puzzle == 0) * candidates[i, :, :] * (certaintyr.reshape(9, 1) + certaintyc)
                change = True

    return puzzle, change

    """

    if change == False:
        masks_sum = np.sum(candidates, axis = 0) == 1

        if np.any(masks_sum):
            change = True
            for i in range(9):
                if completed[i]:
                    certainty = masks_sum * candidates[i, :, :]
                    if np.any(certainty):
                        puzzle += (i+1) * (puzzle == 0) * certainty

    if change == False:
        for i in range(9):
            if completed[i]:
                Mask = candidates[i, :, :]

                certaintyb = np.sum(Mask.reshape(3, 3, 3, 3).swapaxes(1,2).reshape(9, 9), axis=1, keepdims=True)==1
                if np.any(certaintyb):
                    puzzle += (i+1) * (puzzle == 0) * Mask * (np.full(shape=[9, 9], fill_value=True) * certaintyb).reshape(3, 3, 3, 3).swapaxes(1,2).reshape(9,9)
                    change = True
    """

def read_puzzle_branch(puzzle_branches, branch_number):
    branch = abs(puzzle_branches[0, :, :] * (puzzle_branches[1, :, :] <= branch_number))

    guessed_number_mask = (puzzle_branches[0, :, :] < 0) * (puzzle_branches[1, :, :] == branch_number)

    branch = np.where(guessed_number_mask, -branch, branch)

    return branch

def write_puzzle_branch(puzzle, guess_coord, puzzle_branches, branch_number):
    puzzle_branches *= puzzle_branches[1, :, :] < branch_number

    puzzle_branches[1, :, :] += branch_number * (puzzle != 0) * (puzzle_branches[1, :, :] == 0) * (puzzle_branches[0, :, :] == 0)

    puzzle_branches[0, :, :] += puzzle * (puzzle_branches[0, :, :] == 0)
    puzzle_branches[0, guess_coord[0], guess_coord[1]] *= -1  

def branch_predict(puzzle, branch_number, puzzle_branches, went_back, candidates):

    number = 0
    co_ord = (0, 0)

    if went_back:
        branch_number -= 1

        if branch_number == 0:
            return [puzzle, branch_number, puzzle_branches, False]

        puzzle = read_puzzle_branch(puzzle_branches, branch_number)

    while number == 0:

        invalid_candidate = 0

        if went_back:
            i = np.argmin(puzzle)

            co_ord = (int(i/9), i % 9)

            invalid_candidate = -puzzle[co_ord]

            for i in range(invalid_candidate, 9):
                if number == 0:
                    number = (i+1) * coord_constraint(puzzle, co_ord, i+1)

        else:

            masks_sum = np.sum(candidates, axis = 0)

            masks_sum += (masks_sum==0) * 10

            i = np.argmin(masks_sum)

            co_ord = (int(i/9), i % 9)

            for i in range(0, 9):
                if number == 0:
                    number = (i+1) * candidates[i, co_ord[0], co_ord[1]]

        if number == 0:
            went_back = True
            branch_number -= 1

            if branch_number == 0:
                return [puzzle, branch_number, puzzle_branches, False]

            puzzle = read_puzzle_branch(puzzle_branches, branch_number)

    puzzle[co_ord] = number
    write_puzzle_branch(puzzle, co_ord, puzzle_branches, branch_number)
    branch_number += 1

    return [puzzle, branch_number, puzzle_branches, True]

def unique(a):
    unique_repitions = np.bincount((a + np.arange(0 , 90, 10).reshape(9, 1)).ravel(), minlength=90).reshape(9, 10) 
    no_of_unique = np.sum((unique_repitions[:, 1:10]!=0), axis=1) + unique_repitions[:, 0]

    all_unique = myall(no_of_unique == 9)
    return all_unique

def valid_sudoku(puzzle):
    if not unique(puzzle): return False
    if not unique(puzzle.T): return False
    if not unique(puzzle.reshape(3, 3, 3, 3).swapaxes(1,2).reshape(9, 9)): return False

    return True

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

if not valid_sudoku(puzzle):
    print("The puzzle you gave me has no solution")
    sys.exit()

print("\nPuzzle:")
print_puzzle(read_puzzle_branch(puzzle_branches, 0))

for i in range(10):
    puzzle = read_puzzle_branch(puzzle_branches, 0)

    last_puzzle = np.zeros([9, 9], dtype=np.int8)

    candidates = np.full(shape=[9, 9, 9], fill_value=True, dtype=bool)

    puzzle_branches = np.array([puzzle, np.zeros(shape=[9,9])], dtype=np.int8)

    branch_number = 1

    no_of_solutions = 0

    check_other_solutions = False

    still_solveable = True

    while still_solveable and ((no_of_solutions < 1 and not check_other_solutions) or (no_of_solutions <= 1 and check_other_solutions)):

        change = False

        candidates = eliminate_candidates(puzzle, last_puzzle, candidates)

        last_puzzle = puzzle.copy()

        puzzle, change = logical_solve(puzzle, candidates)

        if not change:

            if myall(puzzle > 0):

                    if valid_sudoku(puzzle):
                        no_of_solutions += 1

                        if no_of_solutions == 1:
                            solution = puzzle.copy()

                    puzzle, branch_number, puzzle_branches, still_solveable = branch_predict(puzzle, branch_number, puzzle_branches, True, candidates)

                    last_puzzle = np.zeros([9, 9], dtype=np.int8)
                    candidates = np.full(shape=[9, 9, 9], fill_value=True, dtype=bool)

            else:
                last_branch = branch_number
                puzzle, branch_number, puzzle_branches, still_solveable = branch_predict(puzzle, branch_number, puzzle_branches, False, candidates)

                if last_branch >= branch_number:
                    last_puzzle = np.zeros([9, 9], dtype=np.int8)
                    candidates = np.full(shape=[9, 9, 9], fill_value=True, dtype=bool)

time_elapsed = (time.perf_counter() - start_time) * 1000 / 10

print("Solution:")
print_puzzle(solution)

print("Solved in ", round(time_elapsed, 3), " milliseconds.")

if no_of_solutions > 1:
    print("This isn't a proper Sudoku however, at least one other solution was found.")

if no_of_solutions == 0:
    print("Sorry, this is impossible to solve")

    print("Realised Sudoku can't be solved in", round(time_elapsed, 3), "second(s).")