import numpy as np
import tensorflow as tf
from vae import VariationalAutoencoder
from data import get_data
import argparse


def test_reconstruction(model, tdata, num_samples, h=28, w=28, batch_size=2):
    reconstructed = []
    # Test the trained model: reconstruction
    
    print("Testing the model for reconstruction.\nObtaining reconstructed seq2seq fingerprints \n\n")
    
    print("Reconstructing %d samples ... \n\n" % num_samples)
    
    reconst_counter = 0
    
    for iter in range(num_samples/batch_size):
        
        epoch_input = tdata[iter * batch_size : (iter + 1) * batch_size]
    
        x_reconstructed = model.reconstructor(epoch_input)
        reconstructed.extend(x_reconstructed)
        reconst_counter += batch_size
        
        if reconst_counter % 2 == 0:
            print("Done reconstructing %d/%d lines" % (reconst_counter, num_samples))
        
    print ("\nReconstruction done \n\n")
    
    return reconstructed



if __name__ == "__main__":


	parser = argparse.ArgumentParser()
	parser.add_argument("test_filename", help="path of the file containing SMILES strings for reconstruction")
	parser.add_argument("reconst_filename", help="path of file to write reconstructed strings to, for checking accuracy")

	args = parser.parse_args()
	
	# path of file to write reconstructed strings to for checking accuracy
	#reconst_filename = "smiles_small_reconstructed.fp"

	# Obtain the dataset for testing 
	tdata, num_samples, w, h, input_dim = get_data(args.test_filename)

	# Test the model for reconstruction
	loaded_model = VariationalAutoencoder(input_dim)
	loaded_model.restore_model('./model')

	print("Loaded model from disk")
   
	reconstructed = test_reconstruction(loaded_model, tdata, num_samples)
   
	#writing to a text file
	with open(args.reconst_filename, "w") as File:
		for i,r in enumerate(reconstructed):
			for item in r:
				File.write("%f " % item)
			File.write("\n")

	File.close()