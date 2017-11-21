#sys.test
import os
import sys
for i in range(int(sys.argv[1])):
    os.system("python3 main2-iterative.py" + " " + str(int(sys.argv[2])*(i+1)))