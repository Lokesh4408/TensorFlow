import tensorflow as tf


# examples of different tensors
string = tf.Variable("this is a string", tf.string)
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)

rank1_tensor = tf.Variable(["Test", "Ok", "Tim"], tf.string)
rank2_tensor = tf.Variable([["test", "Ok"], ["test", "yes"], ["test", "yes"], ["test", "yes"]], tf.string)

# to determine the rank of a tensor, Command: tf.rank(tensor_name)
print(tf.rank(rank2_tensor))

print(rank2_tensor.shape)

# changing shape
tensor1 = tf.ones([1, 2, 3])
print(tensor1)
tensor2 = tf.reshape(tensor1, [2,3,1]) # reshape existing data to shape [2,3,1]
tensor3 = tf.reshape(tensor2, [3, -1]) # -1 tells the tensor to calculate the size of the dimension in that place
print(tensor2)                         # this will reshape the tensor to [3,2]
print(tensor3)

# Types of sensors: Variable, Constant, Placeholder, SparseTensor. With the exception of Variable all of these tensors are immutable, meaning their value may not change during execution.
# Evaluating Tensors
with tf.Session() as sess: # creates a session using the default graph
    tensor2.eval() # tensor will of course be the name of your tensor
    
print(tf.__version__)

t = tf.zeros([5,5,5,5])
print(t)
t = tf.reshape(t, [625])
print(t)
t = tf.reshape(t, [125, -1])
print(t)

