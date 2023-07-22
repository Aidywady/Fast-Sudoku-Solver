import numpy as np
import sys
import os
import time

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
    if myany(puzzle[co_ord[0], :] == number): return 0
    if myany(puzzle[:, co_ord[1]] == number): return 0
    a = int(co_ord[0] / 3) * 3
    b = int(co_ord[1] / 3) * 3
    if myany(puzzle[a:a+3, b:b+3] == number): return 0
    return number

def eliminate_candidates(change, candidates):
    candidates *= (change == 0)

    list = np.array(np.nonzero(change))

    if len(list[0, :]) <= 15:
        for i in range(len(list[0, :])):        
            num = change[list[0, i], list[1, i]] - 1

            a = int(list[0, i] / 3) * 3
            b = int(list[1, i] / 3) * 3

            candidates[num, a:a+3, b:b+3]= False
            candidates[num, list[0, i], :] = False 
            candidates[num, :, list[1, i]] = False

        return candidates

    list = np.unique(change)[1:]
    length = len(list)

    number_locations = np.full(shape=[length , 9, 9], fill_value=False)

    for i in range(length):
        number_locations[i, :, :] = (change != list[i])

    row = np.all(number_locations, axis=2, keepdims=True)
    col = np.all(number_locations, axis=1)
    box = np.repeat(np.all(number_locations.reshape(length, 3, 3, 3, 3).swapaxes(2,3).reshape(length, 9, 9), axis=2, keepdims=True), 9).reshape(length, 3, 3, 3, 3).swapaxes(2,3).reshape(length, 9, 9)

    for i in range(length):

        candidates[list[i]-1, :, :] *= row[i, :] * col[i, :] * box[i, :]

    return candidates

def logical_solve(puzzle, candidates):
    change = False

    row_counts = np.sum(candidates, axis=2) == 1
    col_counts = np.sum(candidates, axis=1) == 1

    list = np.any(row_counts + col_counts, axis = 1)

    for i in range(9):
        if list[i]:
            puzzle += (i+1) * (puzzle == 0) * candidates[i, :, :] * (row_counts[i, :].reshape(9, 1) + col_counts[i, :])
            change = True

    return puzzle, change

def read_puzzle_branch(puzzle_branches, branch_number):
    branch = abs(puzzle_branches[0, :, :] * (puzzle_branches[1, :, :] <= branch_number))

    guessed_number_mask = (puzzle_branches[0, :, :] < 0) * (puzzle_branches[1, :, :] == branch_number)
    branch = np.where(guessed_number_mask, -branch, branch)

    return branch

def write_puzzle_branch(puzzle, guess_coord, puzzle_branches, branch_number):
    puzzle_branches *= puzzle_branches[1, :, :] < branch_number

    puzzle_branches[0, :, :] += puzzle * (puzzle_branches[0, :, :] == 0)
    puzzle_branches[0, guess_coord[0], guess_coord[1]] *= -1

    puzzle_branches[1, :, :] += branch_number * (puzzle != 0) * (puzzle_branches[0, :, :] * puzzle_branches[1, :, :] == 0)

def branch_predict(puzzle, branch_number, puzzle_branches, go_back, candidates):

    number = 0
    co_ord = (0, 0)    

    while number == 0:        

        if go_back:         

            branch_number -= 1

            if branch_number == 0:
                return [puzzle, branch_number, puzzle_branches]

            puzzle = read_puzzle_branch(puzzle_branches, branch_number)

            i = np.argmin(puzzle)

            co_ord = (int(i/9), i % 9)

            for i in range(-puzzle[co_ord], 9):
                if number == 0:
                    number = coord_constraint(puzzle, co_ord, i+1)

        else:

            masks_sum = np.sum(candidates, axis = 0)

            masks_sum += (masks_sum==0) * 10

            i = np.argmin(masks_sum)

            co_ord = (int(i/9), i % 9)

            if puzzle[co_ord] == 0:
                for i in range(0, 9):
                    if number == 0:
                        number = (i+1) * candidates[i, co_ord[0], co_ord[1]]

        if number == 0:
            go_back = True

    puzzle[co_ord] = number
    write_puzzle_branch(puzzle, co_ord, puzzle_branches, branch_number)
    branch_number += 1

    return [puzzle, branch_number, puzzle_branches]

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

start_time = time.perf_counter()

data = ''
path = os.getcwd() + '\hard sudoku 002.txt'
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

if not valid_sudoku(puzzle):
    print("The puzzle you gave me has no solution")
    sys.exit()

last_puzzle = np.zeros([9, 9], dtype=np.int8)

candidates = np.full(shape=[9, 9, 9], fill_value=True)

puzzle_branches = np.array([puzzle, np.zeros(shape=[9,9])], dtype=np.int8)

branch_number = 1

no_of_solutions = 0

check_other_solutions = False

print("\nPuzzle:")
print_puzzle(read_puzzle_branch(puzzle_branches, 0))

for i in range(10):
    puzzle = read_puzzle_branch(puzzle_branches, 0)

    last_puzzle = np.zeros([9, 9], dtype=np.int8)

    candidates = np.full(shape=[9, 9, 9], fill_value=True)

    puzzle_branches = np.array([puzzle, np.zeros(shape=[9,9])], dtype=np.int8)

    branch_number = 1

    no_of_solutions = 0

    check_other_solutions = False

    while branch_number > 0 and ((no_of_solutions < 1 and not check_other_solutions) or (no_of_solutions <= 1 and check_other_solutions)):

        candidates = eliminate_candidates(puzzle - last_puzzle, candidates)

        last_puzzle = puzzle.copy()

        puzzle, change = logical_solve(puzzle, candidates)

        if not change:

            if myall(puzzle > 0):

                if valid_sudoku(puzzle):
                    no_of_solutions += 1

                    if no_of_solutions == 1:
                        solution = puzzle.copy()

                puzzle, branch_number, puzzle_branches = branch_predict(puzzle, branch_number, puzzle_branches, True, candidates)

                last_puzzle = np.zeros([9, 9], dtype=np.int8)
                candidates = np.full(shape=[9, 9, 9], fill_value=True)

            else:
                last_branch = branch_number
                puzzle, branch_number, puzzle_branches = branch_predict(puzzle, branch_number, puzzle_branches, False, candidates)

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