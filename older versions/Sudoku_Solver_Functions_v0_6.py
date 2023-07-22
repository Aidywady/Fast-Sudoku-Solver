import numpy as np
import sys
import os

def read_puzzle(filename):
    data = ""
    if not os.path.isfile(filename):
        print("No sudoku puzzle found at", filename)
        sys.exit()
    print("opening puzzle file at path", filename)
    with open(filename, 'r') as file:
        data = file.read()
    data = " ".join(data.replace("-", "0").replace(".", "0").replace("?", "0").replace("*", "0").replace(" ", "").replace("\n", ""))
    if len(data) != 161:
        print("Text file does not contain a valid Sudoku puzzle")
        sys.exit()
    puzzle = np.fromstring(data, sep= " ", dtype=np.int8)
    puzzle = puzzle.reshape(9, 9)
    return puzzle

def read_database(filename):
    data = ''
    length = 0
    if not os.path.isfile(filename):
        print("No sudoku database found at", filename)
        sys.exit()
    print("opening sudoku database at path", filename)
    print("Large databases may take some time read.")
    with open(filename, 'r') as file:
        for ln in file.readlines():
            if not ln.startswith("#"):
                data += " " + " ".join(ln[0:81].replace(".", "0"))
                length += 1

    puzzle = np.fromstring(data, sep= " ", dtype=np.int8)
    puzzle = puzzle.reshape(length, 9, 9)
    return puzzle, length

def write_puzzle(filename, puzzle):
    if isinstance(puzzle, str):
        string = puzzle
    else:
        string = np.array2string(puzzle, separator=' ').replace("[", "").replace("]", "").replace(" ", "").replace("0", ".")
    with open(filename, 'w') as f:
        f.write(string)

def append_to_database(filename, puzzle):
    if isinstance(puzzle, str):
        string = puzzle
    else:
        string = np.array2string(puzzle, separator=' ').replace("[", "").replace("]", "").replace(" ", "").replace("\n", "").replace("0", ".")
    with open(filename, 'a') as f:
        f.write("\n")
        f.write(string)

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

    single_candidates = np.sum(candidates, axis = 0) == 1

    if myany(single_candidates):
        puzzle += single_candidates * np.sum(candidates * np.arange(1, 10).repeat(81).reshape(9, 9, 9), axis=0)
        return puzzle, True

    row_counts = np.sum(candidates, axis=2) == 1
    col_counts = np.sum(candidates, axis=1) == 1

    list = np.any(row_counts + col_counts, axis=1)

    for i in range(9):
        if list[i]:
            puzzle += (i+1) * (puzzle == 0) * candidates[i, :, :] * (row_counts[i, :].reshape(9, 1) + col_counts[i, :])
            change = True

    if change == False:
        box_counts = np.sum(candidates.reshape(9, 3, 3, 3, 3).swapaxes(2,3).reshape(9, 9, 9), axis=2) == 1
        list = np.any(box_counts, axis=1)
        for i in range(9):
            if list[i]:
                puzzle += (i+1) * (puzzle == 0) * candidates[i, :, :] * box_counts[i, :].repeat(9).reshape(3, 3, 3, 3).swapaxes(1,2).reshape(9, 9)
                change = True

    return puzzle, change

def read_puzzle_branch(puzzle_branches, candidate_branches, branch_number):
    branch = abs(puzzle_branches[0, :, :] * (puzzle_branches[1, :, :] <= branch_number))

    guessed_number_mask = (puzzle_branches[0, :, :] < 0) * (puzzle_branches[1, :, :] == branch_number)
    branch = np.where(guessed_number_mask, -branch, branch)

    branch_candidates = (candidate_branches >= branch_number)
    return branch, branch_candidates

def write_puzzle_branch(puzzle, guess_coord, candidates, puzzle_branches, candidate_branches, branch_number):
    puzzle_branches *= puzzle_branches[1, :, :] < branch_number

    puzzle_branches[0, :, :] += puzzle * (puzzle_branches[0, :, :] == 0)
    puzzle_branches[0, guess_coord[0], guess_coord[1]] *= -1

    puzzle_branches[1, :, :] += branch_number * (puzzle != 0) * (puzzle_branches[0, :, :] * puzzle_branches[1, :, :] == 0)

    candidate_branches = np.where(candidate_branches >= branch_number, branch_number - 1, candidate_branches)

    candidate_branches = np.where(candidates, branch_number, candidate_branches)
    return candidate_branches

def branch_predict(puzzle, branch_number, puzzle_branches, candidate_branches, go_back, candidates):

    number = 0
    co_ord = (0, 0)

    while number == 0:

        if go_back:

            branch_number -= 1

            if branch_number == 0:
                return [puzzle, branch_number, puzzle_branches, np.zeros(shape=[9, 9]),  candidates, candidate_branches]

            puzzle, candidates = read_puzzle_branch(puzzle_branches, candidate_branches, branch_number)

            i = np.argmin(puzzle)

            co_ord = (int(i/9), i % 9)

            if np.sum(candidates[:, co_ord[0], co_ord[1]]) <= 1:
                number = np.sum(candidates[:, co_ord[0], co_ord[1]] * np.arange(1, 10, 1))

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

        if number == 0:
            go_back = True

    puzzle[co_ord] = number

    candidates[number - 1, co_ord[0], co_ord[1]] = False

    candidate_branches = write_puzzle_branch(puzzle, co_ord, candidates, puzzle_branches, candidate_branches, branch_number)
    branch_number += 1

    last_puzzle = np.zeros(shape=[9, 9], dtype=np.int8)

    last_puzzle[co_ord] = number

    candidates = eliminate_candidates(last_puzzle, candidates)
    last_puzzle = puzzle.copy()

    return [puzzle, branch_number, puzzle_branches, last_puzzle, candidates, candidate_branches]

def random_guess(puzzle, branch_number, puzzle_branches, candidate_branches, go_back, candidates):

    number = 0
    co_ord = (0, 0)

    while number == 0:

        temp = 0

        if go_back:

            branch_number -= 1

            puzzle, candidates = read_puzzle_branch(puzzle_branches, candidate_branches, branch_number)

        temp = np.argmin(puzzle)

        co_ord = (int(temp/9), temp % 9)

        random_sequence = np.random.permutation(np.arange(1, 10, 1) * candidates[:, co_ord[0], co_ord[1]])

        for i in range(0, 9):
            if number == 0:
                number = random_sequence[i]

        if number == 0:
            go_back = True

    puzzle[co_ord] = number

    candidates[number - 1, co_ord[0], co_ord[1]] = False

    candidate_branches = write_puzzle_branch(puzzle, co_ord, candidates, puzzle_branches, candidate_branches, branch_number)
    branch_number += 1

    last_puzzle = np.zeros(shape=[9, 9], dtype=np.int8)

    last_puzzle[co_ord] = number

    candidates = eliminate_candidates(last_puzzle, candidates)
    last_puzzle = puzzle.copy()

    return [puzzle, branch_number, puzzle_branches, last_puzzle, candidates, candidate_branches]

def unique(a):
    return myall(np.bincount(a.ravel() + np.arange(0, 90, 10).repeat(9), minlength=90).reshape(9, 10)[:, 1:10] <= 1)

def valid_sudoku(puzzle):
    if not unique(puzzle): return False
    if not unique(puzzle.T): return False
    if not unique(puzzle.reshape(3, 3, 3, 3).swapaxes(1,2)): return False

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

def Solve(puzzle, check_other_solutions):
    if not valid_sudoku(puzzle):
        return np.zeros(shape=[9, 9], dtype=np.int8), 0, 0

    if len(np.unique(puzzle)) <= 8:

        return np.zeros(shape=[9, 9], dtype=np.int8), 2, 0

    solution = np.zeros(shape=[9, 9], dtype=np.int8)

    last_puzzle = np.zeros([9, 9], dtype=np.int8)

    candidates = np.full(shape=[9, 9, 9], fill_value=True)

    puzzle_branches = np.array([puzzle, np.zeros(shape=[9,9])], dtype=np.int8)

    candidate_branches = np.zeros(shape=[9, 9, 9], dtype=np.int8)

    candidates = eliminate_candidates(puzzle - last_puzzle, candidates)

    last_puzzle = puzzle.copy()

    no_of_guesses = 0
    final_no_of_guesses = 0

    branch_number = 1

    no_of_solutions = 0

    while branch_number > 0 and ((no_of_solutions < 1 and not check_other_solutions) or (no_of_solutions <= 1 and check_other_solutions)):

        puzzle, change = logical_solve(puzzle, candidates)

        if change:
            candidates = eliminate_candidates(puzzle - last_puzzle, candidates)

            last_puzzle = puzzle.copy()

        if not change:
            go_back = False

            if valid_sudoku(puzzle):

                if myall(puzzle > 0):
                    no_of_solutions += 1

                    if no_of_solutions == 1:
                        solution = puzzle.copy()
                        final_no_of_guesses = no_of_guesses
                    go_back = True
            else:
                go_back = True

            no_of_guesses += 1
            puzzle, branch_number, puzzle_branches, last_puzzle, candidates, candidate_branches = branch_predict(puzzle, branch_number, puzzle_branches, candidate_branches, go_back, candidates)

    return solution, no_of_solutions, final_no_of_guesses

def Generate():
    puzzle = np.zeros(shape=[9, 9], dtype=np.int8)

    last_puzzle = np.zeros([9, 9], dtype=np.int8)

    candidates = np.full(shape=[9, 9, 9], fill_value=True)

    puzzle_branches = np.array([puzzle, np.zeros(shape=[9,9])], dtype=np.int8)

    candidate_branches = np.zeros(shape=[9, 9, 9], dtype=np.int8)

    candidates = eliminate_candidates(puzzle - last_puzzle, candidates)

    last_puzzle = puzzle.copy()

    branch_number = 1

    while myany(puzzle == 0):

        puzzle, change = logical_solve(puzzle, candidates)

        if change:
            candidates = eliminate_candidates(puzzle - last_puzzle, candidates)

            last_puzzle = puzzle.copy()

        if not change:
            go_back = False

            if not valid_sudoku(puzzle):
                go_back = True

            puzzle, branch_number, puzzle_branches, last_puzzle, candidates, candidate_branches = random_guess(puzzle, branch_number, puzzle_branches, candidate_branches, go_back, candidates)

    return puzzle

def generate_minimal_puzzle(solution):
    permutation = np.random.permutation(np.arange(0, 81, 1))

    mask = np.full(shape=[9, 9], fill_value=True)

    no_of_solutions = 2

    for i in permutation:
        co_ord = (int(i/9), i % 9)

        mask[co_ord] = False

        temp, no_of_solutions, no_of_guesses = Solve(solution * mask, True)

        if no_of_solutions != 1:
            mask[co_ord] = True

    puzzle = solution * mask

    temp, no_of_solutions, no_of_guesses = Solve(puzzle.copy(), True)

    no_of_clues = np.count_nonzero(puzzle)

    difficulty = ""

    if no_of_guesses == 0:
        difficulty = "very easy"

    if no_of_guesses > 0 and no_of_guesses <= 5:
        difficulty = "easy"

    if no_of_guesses > 5 and no_of_guesses <= 10:
        difficulty = "medium"

    if no_of_guesses > 10:
        difficulty = "hard"

    return puzzle, no_of_clues, difficulty