import pickle
import numpy as np

matrix = [[1, 0, 0, 0],
	[0, 1, 0, 0],
	[0, 0, 1, 0]]

f = open("endoscope_to_psm2", "w+")
# matrix.tofile(f)
pickle.dump(matrix, f)
f.close()

# g = open("psm2_to_endoscope", "rb")
# print pickle.load(g)
# # print np.fromfile(g)
# g.close()
