import time
import numpy as np
import tensorflow as tf
from vae import VariationalAutoencoder
from data import get_data
import argparse

def trainer(model_object, tdata, num_samples, input_dim, learning_rate=1e-4, batch_size=2, num_epoch=10, n_z=16, log_step=2):
    

	model = model_object(input_dim, learning_rate=learning_rate, batch_size=batch_size, n_z=n_z)
            
        #saver = tf.train.Saver()
	print("Training .... \n\n")
    
	for epoch in range(num_epoch):
		start_time = time.time()
            # Get a batch
            # Execute the forward and backward pass 
            # Report computed losses
		for iter in range( num_samples/batch_size):
			epoch_input = tdata[iter * batch_size : (iter + 1) * batch_size]
			losses = model.run_single_step(epoch_input)
		end_time = time.time()
        
            #saver.save(sess=model.sess, save_path='./my_test_model.ckpt')
        
		if epoch % log_step == 0:
			log_str = '[Epoch {}] '.format(epoch)
			for k, v in losses.items():
				log_str += '{}: {:.3f}  '.format(k, v)
			log_str += '({:.3f} sec/epoch)'.format(end_time - start_time)
			print(log_str)
                    
	print('\nTraining done!\n\n')
    
	model.save_model('./model')
	print("Saved model to disk\n\n")
    
	return model



if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("train_filename", help="path of text file containing SMILES strings for training")
	args = parser.parse_args()

    # path of text file containing SMILES strings for training and testing
    #train_filename = '../data/smiles_small.fp'
    
    # Obtain the dataset for training 
	tdata, num_samples, w, h, input_dim = get_data(args.train_filename)
    
    # Train a model
	model = trainer(VariationalAutoencoder, tdata, num_samples, input_dim)