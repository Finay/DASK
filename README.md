# Team Project

Please keep this up-to-date with information about your project throughout the term.

The readme should include information such as:
- a summary of what your application is all about
- a list of the user stories, along with who is responsible for each one
- information about the API(s) that your project uses 
- screenshots or animations demonstrating current functionality

By keeping this README up-to-date,
your team will find it easier to prepare for the final presentation
at the end of the term.

# Infodump
Our tensor is a multidimensional matrix of floats supporting a variety of mathematical operations with corresponding gradient 
propagation methods.

To support operations being performed on a variety of devices, a tensor object stores its data and calls operations 
through Storage and Backend interfaces. 

Let's say we want to matrix multiply a nxm matrix by a mxl matrix. 
1. We initialize the first tensor by passing an array of floats of length n*m, the desired shape of the matrix, and the device operations will be performed on to the tensor constructor.
   1. `StorageFactory` will create storage on the specified device to store the input floats.
2. The second tensor is created in the same way.
3. Use the `TensorOps.matmul` static method on the two matrices.
   1. A Backend for the input tensor's device is created by the `BackendFactory`.
   2. The Backend performs the calculations for matmul and produces a result tensor.
   3. The backwards propagation method of the result tensor is set to propagate its gradients to the input tensors.
   4. Result tensor is returned.
 