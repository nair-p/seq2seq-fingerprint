
# coding: utf-8

# In[1]:


import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
#%matplotlib inline


# In[15]:


class VariationalAutoencoder(object):

    def __init__(self, input_dim, learning_rate=1e-4, batch_size=2, n_z=16):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_z = n_z

        tf.reset_default_graph()
        self.build()

        self.sess = tf.InteractiveSession()
        #self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        

    # Build the netowrk and the loss functions
    def build(self):
        self.x = tf.placeholder(
            name='x', dtype=tf.float32, shape=[None, input_dim])

        # Encode
        # x -> z_mean, z_sigma -> z
        f1 = fc(self.x, 256, scope='enc_fc1', activation_fn=tf.nn.relu)
        f2 = fc(f1, 128, scope='enc_fc2', activation_fn=tf.nn.relu)
        f3 = fc(f2, 64, scope='enc_fc3', activation_fn=tf.nn.relu)
        self.z_mu = fc(f3, self.n_z, scope='enc_fc4_mu', 
                       activation_fn=None)
        self.z_log_sigma_sq = fc(f3, self.n_z, scope='enc_fc4_sigma', 
                                 activation_fn=None)
        eps = tf.random_normal(
            shape=tf.shape(self.z_log_sigma_sq),
            mean=0, stddev=1, dtype=tf.float32)
        self.z = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps

        # Decode
        # z -> x_hat
        g1 = fc(self.z, 64, scope='dec_fc1', activation_fn=tf.nn.relu)
        g2 = fc(g1, 128, scope='dec_fc2', activation_fn=tf.nn.relu)
        g3 = fc(g2, 256, scope='dec_fc3', activation_fn=tf.nn.relu)
        self.x_hat = fc(g3, input_dim, scope='dec_fc4', 
                        activation_fn=tf.sigmoid)

        # Loss
        # Reconstruction loss
        # Mean-squared error loss
        self.recon_loss = tf.reduce_mean(tf.squared_difference(self.x, self.x_hat))
        
        # Latent loss
        # KL divergence: measure the difference between two distributions
        # Here we measure the divergence between 
        # the latent distribution and N(0, 1)
        latent_loss = -0.5 * tf.reduce_sum(
            1 + self.z_log_sigma_sq - tf.square(self.z_mu) - 
            tf.exp(self.z_log_sigma_sq), axis=1)
        self.latent_loss = tf.reduce_mean(latent_loss)

        self.total_loss = self.recon_loss + self.latent_loss
        
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.total_loss)
        
        self.losses = {
            'recon_loss': self.recon_loss,
            'latent_loss': self.latent_loss,
            'total_loss': self.total_loss,
        }        
        return

    # Execute the forward and the backward pass
    def run_single_step(self, x):
        _, losses = self.sess.run(
            [self.train_op, self.losses],
            feed_dict={self.x: x}
        )
        return losses

    # x -> x_hat
    def reconstructor(self, x):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x})
        return x_hat

    # z -> x
    def generator(self, z):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.z: z})
        return x_hat
    
    # x -> z
    def transformer(self, x):
        z = self.sess.run(self.z, feed_dict={self.x: x})
        return z
    
    # function to save model
    def save_model(self, save_path):
        saver = tf.train.Saver()
        saver.save(self.sess, save_path)
        
    # function to restore model
    def restore_model(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)


# In[10]:


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


# In[4]:


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


# In[5]:


def get_train_data(filepath):
    
    # list to hold the string values 
    data = []
    input_dim = 0
    
    # reading text file for training data
    with open(train_filename, 'r') as File:
        infoFile = File.readlines() #Reading all the lines from File
        total_lines = len(infoFile)
        print("Reading %d lines from training file\n\n"%total_lines)

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


# In[11]:


if __name__ == "__main__":
    
    # path of text file containing SMILES strings for training and testing
    train_filename = '../data/smiles_small.fp'
    
    # path of file to write reconstructed strings to for checking accuracy
    reconst_filename = "smiles_small_reconstructed.fp"
    
    # Obtain the dataset for training 
    tdata, num_samples, w, h, input_dim = get_train_data(train_filename)
    
    # Train a model
    model = trainer(VariationalAutoencoder, tdata, num_samples, input_dim)
    
    #saver = tf.train.Saver()
    
    # serialize model to JSON
    #model_json = model.to_json()
    #with open("model_small.json", "w") as json_file:
    #    json_file.write(model_json)
    # serialize weights to HDF5
    #model.save_weights("model.h5")
    #saver.save(sess=model.sess, save_path='./my_test_model.ckpt')
    #model.save_model_to_dir("./", sess=model.sess)
    
    #with open("./model.ckpt", "wb")
    
   
    # load json and create model
    #json_file = open('model.json', 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    #loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    #loaded_model.load_weights("model.h5")
    #saver.restore(sess=model.sess, './my_test_model.ckpt')
    #print("Loaded model from disk")
    
    #reconstructed = test_reconstruction(loaded_model, tdata, num_samples)
    
    # writing to a text file
    #with open(reconst_filename, "w") as File:
    #    for i,r in enumerate(reconstructed):
    #        for item in r:
    #            File.write("%f " % item)
    #        File.write("\n")

    #File.close()


# In[20]:


# Test the model for reconstruction
loaded_model = VariationalAutoencoder(input_dim)
loaded_model.restore_model('./model')
print("Loaded model from disk")
   
reconstructed = test_reconstruction(loaded_model, tdata, num_samples)
   
#writing to a text file
with open(reconst_filename, "w") as File:
   for i,r in enumerate(reconstructed):
       for item in r:
           File.write("%f " % item)
       File.write("\n")

File.close()

