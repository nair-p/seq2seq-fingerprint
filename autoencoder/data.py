### This file contains the function to read data points from a file which is passed as an argument to it
### This function returns the read data samples, the number of samples (size of dataset), dimension of weight and hidden matrices
### It also returns the input dimension

# To use while training, call this function with the training data file path as argument

# To use while testing, call this function with the test data file path as argument

import numpy as np

def get_data(filepath):
    
    # list to hold the string values 
	data = []
	input_dim = 0
    
    # reading text file for training data
	with open(filepath, 'r') as File:
		infoFile = File.readlines() #Reading all the lines from File
		total_lines = len(infoFile)
		print("Reading %d lines from file\n\n"%total_lines)

		line_number = 0
        
		for line in infoFile: #Reading line-by-line
            
			line_number += 1
			l = line[:-1].split()
			tmp = []
			
			for item in l:
				tmp.append(float(item))
			data.append(tmp)
            
			if line_number % 2 == 0:
				print("Done with %d/%d lines"%(line_number, total_lines))
                
			input_dim = max(input_dim, len(line.split()))

	num_sample = len(data)
	w = h = int(np.sqrt(input_dim))
    
	print("\nData reading done\n\n")
	return data, num_sample, w, h, input_dim