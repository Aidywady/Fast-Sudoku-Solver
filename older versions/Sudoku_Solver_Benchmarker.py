import numpy as np
import os
import time

import Sudoku_Solver_APIs as sudoku
import Sudoku_Solver_APIs_v0_6 as test



#Benchmarker

start_time = time.perf_counter()

easy_path = os.getcwd() + '\\puzzles\\databases\\UnbiasedDataset.txt'
medium_path = os.getcwd() + '\\puzzles\\databases\\ModeratelyHardDataset.txt'
hard_path = os.getcwd() + '\\puzzles\\databases\\10.0+RatedDataset.txt'
hardest_path = os.getcwd() + '\\puzzles\\databases\\HardestDataset.txt'

easy_puzzles, database1_length = sudoku.read_database(easy_path)
medium_puzzles, database2_length = sudoku.read_database(medium_path)
hard_puzzles, database3_length = sudoku.read_database(hard_path)
hardest_puzzles, database4_length = sudoku.read_database(hardest_path)

easy_puzzles.resize(1000, 9, 9)
medium_puzzles.resize(1000, 9, 9)
hard_puzzles.resize(1000, 9, 9)
hardest_puzzles.resize(1000, 9, 9)

database_lengths = np.array((database1_length, database2_length, database3_length, database4_length))
merged_databases = np.concatenate((easy_puzzles, medium_puzzles, hard_puzzles, hardest_puzzles)).reshape(4, 1000, 9, 9)

database_lengths[database_lengths>1000] = 1000 

time_elapsed = (time.perf_counter() - start_time) * 1000

print("Databases read in", time_elapsed, "ms")

database_times_files = ['easy_database_times.csv', 
                        'medium_database_times.csv', 
                        'hard_database_times.csv',
                        'hardest_database_times.csv']

with open(database_times_files[0], 'w') as f:
    f.write("# Times for easy puzzles")
    
with open(database_times_files[1], 'w') as f:
    f.write("# Times for medium puzzles")
    
with open(database_times_files[2], 'w') as f:
    f.write("# Times for hard puzzles")
    
with open(database_times_files[3], 'w') as f:
    f.write("# Times for hardest puzzles")

for i in range(4):
	for number in range(database_lengths[i]):
		puzzle = merged_databases[i, number, :, :]
    
		print("Puzzle", number+1, "/", database_lengths[i], ":", end=' ')

		start_time = time.perf_counter()

		solution, no_of_solutions = test.Solve(puzzle.copy(), False)

		time_elapsed = (time.perf_counter() - start_time) * 1000

		if np.all(solution > 0) and np.all(solution * (puzzle != 0) == puzzle) and sudoku.valid_sudoku(solution):
			print("Solved.")
			with open(database_times_files[i], 'a') as f:
				f.write('\n')
				f.write(str(round(time_elapsed, 3)))
		else:
			print("Error in algorithm.")
			with open(database_times_files[i], 'a') as f:
				f.write('\nInvalid result.')
    	
print("Completed benchmarking!")