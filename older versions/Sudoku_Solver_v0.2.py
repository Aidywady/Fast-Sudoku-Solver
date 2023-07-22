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

candidates = np.full(shape=[9, 9, 9], fill_value=True, dtype=bool)

puzzle_branches = np.array([puzzle, np.zeros(shape=[9,9])], dtype=np.int8)

branch_number = 1

no_of_solutions = 0

check_other_solutions = False

still_solveable = True

def generate_constraints(puzzle, number): 

    number_locations = puzzle != number
    Mask = np.tile(np.all(number_locations, axis=1, keepdims=True), [1,9]) 
    Mask *= np.tile(np.all(number_locations, axis=0, keepdims=True), [9,1])
    Mask *= np.tile(np.all(number_locations.reshape(3, 3, 3, 3).swapaxes(1,2).reshape(9, 9), axis=1, keepdims=True), [1,9]).reshape(3, 3, 3, 3).swapaxes(1,2).reshape(9,9)

    return Mask

def coord_constraint(puzzle, co_ord, number):
    if puzzle[co_ord] > 0: return False
    if np.any(puzzle[co_ord[0], :] == number): return False
    if np.any(puzzle[:, co_ord[1]] == number): return False
    if np.any(puzzle.reshape(3, 3, 3, 3).swapaxes(1,2).reshape(9, 9)[(int(co_ord[0] / 3)*3 +int(co_ord[1]/3)), :] == number): return False
    return True   

def unique(a):
    a_offset = a + np.tile(np.arange(0 , 90, 10).reshape(9, 1), [1, 9])
    unique_repitions = np.bincount(a_offset.ravel(), minlength=90).reshape(9, 10) 
    no_of_unique = (unique_repitions[:, 1:10]!=0).sum(axis=1) + unique_repitions[:, 0]

    all_unique = np.all(no_of_unique == 9)
    return all_unique

def valid_sudoku(puzzle):
    if not unique(puzzle): return False
    if not unique(puzzle.T): return False
    if not unique(puzzle.reshape(3, 3, 3, 3).swapaxes(1,2).reshape(9, 9)): return False

    return True

def solve_step(puzzle):

    change = False
    masks_sum = np.zeros([9,9])

    for i in range(9):
        if np.count_nonzero(puzzle==i+1) < 9:
            Mask = generate_constraints(puzzle, i+1) * (puzzle == 0)
            masks_sum += Mask

            certainty = np.count_nonzero(Mask, axis=1, keepdims=True)==1
            if np.any(certainty):
                puzzle += (i+1) * Mask * np.tile(certainty, [1,9])
                change = True

            certainty = np.count_nonzero(Mask, axis=0, keepdims=True)==1   
            if np.any(certainty):
                puzzle += (puzzle == 0) * (i+1) * Mask * np.tile(certainty, [9,1])
                change = True

            certainty = np.count_nonzero(Mask.reshape(3, 3, 3, 3).swapaxes(1,2).reshape(9, 9), axis=1, keepdims=True)==1
            if np.any(certainty):
                puzzle += (puzzle == 0) * (i+1) * Mask * np.tile(certainty, [1,9]).reshape(3, 3, 3, 3).swapaxes(1,2).reshape(9,9)
                change = True

    return puzzle, change, masks_sum

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

def branch_predict(puzzle, branch_number, puzzle_branches, masks_sum, went_back):

    number = 0
    co_ord = (0, 0)

    while number == 0:

        invalid_candidate = 0

        if went_back:
            i = np.argmin(puzzle)

            co_ord = (int(i/9), i % 9)

            invalid_candidate = -puzzle[co_ord]

        else:

            masks_sum += (masks_sum==0) * 10

            i = np.argmin(masks_sum)

            co_ord = (int(i/9), i % 9)

        for i in range(invalid_candidate, 9):
            if number == 0:
                number = (i+1) * coord_constraint(puzzle, co_ord, i+1)

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

            print(str(puzzle[row][col]), end=" ")

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

    candidates = np.full(shape=[9, 9, 9], fill_value=True, dtype=bool)

    puzzle_branches = np.array([puzzle, np.zeros(shape=[9,9])], dtype=np.int8)

    branch_number = 1

    no_of_solutions = 0

    check_other_solutions = False

    still_solveable = True
    while still_solveable and ((no_of_solutions < 1 and not check_other_solutions) or (no_of_solutions <= 1 and check_other_solutions)):

        change = False

        puzzle, change, masks_sum = solve_step(puzzle)

        if not change:

            if np.all(puzzle > 0):

                if valid_sudoku(puzzle): 
                    no_of_solutions += 1

                    if no_of_solutions == 1:
                        solution = puzzle.copy()

                branch_number -= 1

                if branch_number == 0:
                   still_solveable = False

                else:
                    puzzle = read_puzzle_branch(puzzle_branches, branch_number)

                    puzzle, branch_number, puzzle_branches, still_solveable = branch_predict(puzzle, branch_number, puzzle_branches, masks_sum, True)

            else:
                puzzle, branch_number, puzzle_branches, still_solveable = branch_predict(puzzle, branch_number, puzzle_branches, masks_sum, False)

time_elapsed = (time.perf_counter() - start_time) * 1000 / 10

print("Solution:")
print_puzzle(solution)

print("Solved in ", round(time_elapsed, 3), " milliseconds.")

if no_of_solutions > 1:
    print("This isn't a proper Sudoku however, at least one other solution was found.")

if no_of_solutions == 0:
    print("Sorry, this is impossible to solve")

    print("Realised Sudoku can't be solved in", round(time_elapsed, 3), "second(s).")