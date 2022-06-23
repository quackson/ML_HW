Code from ["Towards deep learning with segregated dendrites" by Jordan Guergiuev, Timothy P. Lillicrap, Blake A. Richards.](https://arxiv.org/abs/1610.00161)

See `example_script.py` for an example of loading MNIST, creating a network and training it.

This code requires the [MNIST dataset files](http://yann.lecun.com/exdb/mnist/) to be downloaded and placed in the same directory as the code.


-----------------------------------------
Ours README.md:

directory structures:
`train.py` loads MNIST, creates a network and trains the model
`basenode.py` defines BaseNode for our neurons
`models.py` defines the neurons and networks
`utils.py` contains utilization functions
`params.py` contains parameters used for this model
`MNIST.py` contains functions that preprocess the MNIST dataset