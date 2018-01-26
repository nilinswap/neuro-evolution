#sys.test
import os
import sys

for i in range(int(sys.argv[1])):
    os.system("python3 main_just_src.py")


"""
	logf = open("log.txt", "a")

	try:
		test_it_with_bp(play = 1, NGEN = 100, MU = 4*25, play_with_whole_pareto = 1)
	except Exception as e:
		logf.write(str(e)+'\n')
	finally:
		logf.close()
		print("ERROR! ERROR! ERROR!")
"""