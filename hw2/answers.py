r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**

**A** The shape of $\frac{\partial Y}{\partial X}$ is $(512\cdot N, 1024\cdot N) = (512\cdot 64, 1024\cdot 64) = (32768 , 65536)$

**B** It's shape is $(64, 64)$, The Jacobian is also block-diagonal, meaning that all blocks that aren't on the main diagonal have only 0 as values. This is because each sample in the batch is not influenced by any other samples. meaning that for each element in the block matrix: $j\neq i \rightarrow \frac{\partial y_i}{\partial x_j}$
Also, because all predictions in this batch are calculated using the same weights, every block on the diagonal is equal to the weights matrix.

**C** Instead of storing a huge $(512\cdot N, 1024\cdot N)$ matrix in memory, we can use the observations we already made to use only the weights matrix. the new tensor has the same shape as the weights matrix $(512, 1024)$

**D** To calculate the downstream gradient w.r.t. the input withour materializing the Jacobian, we can calculate it using a simple matrix multiplication per batch. using $Y=XW^T$ we can show that $\delta X = \delta Y \cdot W$.

**E** The shape of the tensor is $(512 \cdot 64, 512 \cdot 1024)$. The shape of the blocks are the same as before, we a block size of $(512, 1024)$.

```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:** The second order derivative can provide us with useful information, specifically the slope of the gradient. meaning we can use it to specify the step size with each gradient step. it will help us traverse a better route than with using a predefined step size.


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.5, 0.01, 0
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0.1,
        0.02,
        0.003,
        0,
        0.001,
    )
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd, lr, = (
        0.1,
        0.001,
    )
    # ========================
    return dict(wstd=wstd, lr=lr)

# TODO: fix part2_q1
part2_q1 = r"""
**Your answer:**

### Question 1 

Regarding the graphs you got for the three dropout configurations:

1. Explain the graphs of no-dropout vs dropout. Do they match what you expected to see?
    - If yes, explain why and provide examples based on the graphs.
    - If no, explain what you think the problem is and what should be modified to fix it.

2. Compare the low-dropout setting to the high-dropout setting and explain based on your graphs.

1. The graphs match our expectations. Dropout is a method intended to improve the generalization of our model. With dropout=0, we can see the model is severely overfit (train_acc=0.7, test_acc=0.5). We can observe that setting dropout=0.4 has fixed the problem (train_acc=0.5=test_acc) the model performs worse on the training set but the overfitting problem has been mostly solved. of course setting the dropout=0.8 is too high and the model has a hard time learning when most of its weights are being erased while training. 

while training with dropout=0.4, the model doesn't retain subtle training set data. weights that have "lived" for a long time and had a chance to "memorize" the train set also have a high probability to be dropped, this could explain the observed behaviour.

2. the high dropout graph features a very low accuracy and a very high loss curve. this is because the model keeps "forgetting" most of its weights, it can't perform well on the train set or on the test set. The low dropout setting features graphs of a very obviously overfitted model, this is because the model has perfect "memory" and it has learned the train set. the mid dropout=0.4 is a perfect compromise, the model cant memorize the train set but can learn its underlying distribution well.

"""

part2_q2 = r"""
**Your answer:**


It is possible. A scenario that could happen is that the model decreases the loss dramatically for a single sample while increasing the loss slightly for a few other models (just enough to change the prediction). this means that the accuracy will decrease (because the model will be wrong for the few samples) and the total loss will decrease. this scenario is possible in a single epoch or in many epochs.

"""

part2_q3 = r"""
**Your answer:**


1. GD calculates the gradient of loss on the entire dataset for each step. SGD calculates the gradient of loss on parts of the dataset (batches) for each step. SGD takes less computing power and is more efficient than GD. calculating the gradients in SGD makes it possible to escape local minima, since the gradients of different batches may show different local minima.

2. The advantages of momentum will be beneficial for GD as well. since it can make the convergence direction more consistent and accelerate the convergence. especially in cases of large difference in the partial derivatives (shallow in one axis and very deep in another)

3. 
a - Yes, we will achieve the same gradient. because in GD the Loss is defined as the mean of all individual sample loss, which is the same as the mean of all batch losses:
$L_{GD}(w) = \frac{1}{N} \sum^N_{i=1}L_i(w) = \frac{1}{N_{batches}} \sum^{N_{batches}}_{i=1} \sum^{N_{batch}}_{j=1} L_{ij}(w)= L_{GD}^*(w)$

b - We make lots of consecutive forward passes. but the values we save for each batch (to cache for gradient calculation) are still stored in memory. this means that while we save memory on the samples themselves, the cached data would still be too large.

c - To fix this issue, we could accumulate the gradients. after each forward pass calculate gradients with a backward pass and accumulate. after accumulating all the gradients we can execute the gradient step. this means that instead of caching all the data in the network, we can just save the accumulated gradients in memory. this is the same as regular GD, since the weights aren't changed between different batches.

"""


# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    
    n_layers = 3
    hidden_dims = 6
    activation = "relu"
    out_activation = "softmax"
    
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.functional.cross_entropy  # One of the torch.nn losses
    lr, weight_decay, momentum = 0.005, 0.001, 0.9  # Arguments for SGD optimizer
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**


1.
    1. optimization error is the difference between the minimal theoretical error achievable by an optimal model on the training set and the error achieved by training.
    2. generalization error is a measure of how accurately our model can predict samples that are not featured in it's training set
    3. approximation error represents the difference between the true label of a sample and the models prediction for it.

2. As we can see in the accuracy graphs, we have a certain optimization error (only 0.45 loss on training set), but the model is not very overfit. A certain optimization error will always show and we find this one to be acceptable. 
The generalization error is quite high since we only get ~82% accuracy on the test set. This indicates a certain level of overfitting which is quite high.
The approximation error is also acceptable since we reached about 0.46 loss on the test set
"""

part3_q2 = r"""
**Your answer:**


an example scenario where we would prefer to optimize for FPR (False positive) at the cost of increasing FNR (False negative) is for a model that predicts if patients have a deadly disease, for which the medicine is very cheap and has little to no side-effects. We would rather many healthy people to take medicine they don't need than for a few to go without medicine they desperately need.

an example for the reverse would be the same but for a mild disease with an expensive medicine. we wouldn't want people to needlessly buy the medicine, and we wouldn't mind not treating people as much.

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()  # One of the torch.nn losses
    lr, weight_decay, momentum = 0.03, 0, 0  # Arguments for SGD optimizer
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**

1. **Number of Parameters:**
    $C=256$ the number of input and output channels.
   - **Regular Block:** Consists of two $3 \times 3$ convolutions with $C$ filters each.<br>
   Total parameters: $2 \times (3 \times 3 \times C \times C) = 2 \times (9 \times 256^2) = 1,179,648$.
   - **Bottleneck Block:** 
     - Layer 1 ($1 \times 1$): $1 \times 1 \times 256 \times 64 = 16,384$
     - Layer 2 ($3 \times 3$): $3 \times 3 \times 64 \times 64 = 36,864$
     - Layer 3 ($1 \times 1$): $1 \times 1 \times 64 \times 256 = 16,384$<br>
     Total parameters: $16,384 + 36,864 + 16,384 = 69,632$.

2. **Floating Point Operations:**
   The number of FLOPs in a convolutional layer is calculated as:  
   $\text{FLOPs} \approx \text{Parameters} \times (\text{Output Height} \times \text{Output Width})$.  
   
   both the regular block and the bottleneck block are designed to preserve the spatial dimensions of the input (using padding where necessary). Since both blocks operate on the same feature map resolution ($H \times W$), the difference in computational cost depends solely on the number of parameters in each block.  
 
    $$\frac{\text{Parameters}_{\text{Regular}}}{\text{Parameters}_{\text{Bottleneck}}} = \frac{1,179,648}{69,632} \approx 16.94$$

   This means the bottleneck block is approximately 17 times more efficient in terms of FLOPs than the regular block for the same number of input/output channels

3. **Ability to Integrate Input:**
   - **Spatially:** The regular block uses two $3 \times 3$ layers, resulting in a $5 \times 5$ receptive field. The bottleneck block uses only one $3 \times 3$ layer, resulting in a smaller $3 \times 3$ receptive field. Thus, the regular block has a better capacity for spatial integration.
   - **Across Feature Maps:** The bottleneck block utilizes two $1 \times 1$ convolutions specifically designed for channel mixing. This allows for more complex inter-channel combinations and non-linearities (as each conv is followed by an activation) despite the lower parameter count.

"""


part4_q2 = r"""
**Your answer:**

1. For $y_1 = \mat{M} \vec{x}_1$, the gradient with respect to the input is:
   $$\frac{\partial L}{\partial \vec{x}_1} = \frac{\partial L}{\partial \vec{y}_1} \cdot \frac{\partial \vec{y}_1}{\partial \vec{x}_1} = \frac{\partial L}{\partial \vec{y}_1} \mat{M}$$
   In a very deep network, the gradient is multiplied by $\mat{M}$ at each layer. If the singular values of $\mat{M}$ are small, the gradient will decay exponentially (Vanishing Gradient).

2. For $\vec{y}_2 = \vec{x}_2 + \mat{M} \vec{x}_2 = (\mat{I} + \mat{M}) \vec{x}_2$, the gradient is:
   $$\frac{\partial L}{\partial \vec{x}_2} = \frac{\partial L}{\partial \vec{y}_2} \cdot \frac{\partial \vec{y}_2}{\partial \vec{x}_2} = \frac{\partial L}{\partial \vec{y}_2} (\mat{I} + \mat{M})$$

3. The presence of the Identity matrix $\mat{I}$ in the residual gradient $(\mat{I} + \mat{M})$ ensures that the gradient can flow directly through the skip connection even if the learned weights $\mat{M}$ are very small or cause the gradient to vanish. This creates a "highway" for the gradient, allowing it to reach earlier layers in the network without being significantly diminished, which facilitates the training of extremely deep architectures.

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
