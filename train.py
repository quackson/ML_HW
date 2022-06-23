import params
from models import Guerguiev2017Network

# silence 80% of feedback weights
params.use_sparse_feedback = True

# set training parameters
f_etas = (0.21, 0.21, 0.21)
b_etas = None
n_epochs = 60
n_training_examples = 60000

# create the network -- this will also load the MNIST dataset files
net = Guerguiev2017Network(n=(2000,500,10))

# train the network
net.train(f_etas, b_etas, n_epochs, n_training_examples, save_simulation=True, simulations_folder="Simulations", folder_name="Example Simulation")
