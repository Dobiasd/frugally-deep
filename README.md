frugally-deep
=============

**is just a bunch of random code, intended solely for my personal learning pleasure.**

The goal is that someday the following things will be true:

![logo](logo/frugally_deep.jpg)

**frugally-deep**

* **is a header-only deep learning framework written in C++.**
* is frugal because it is minimalistic and tries to consume not so much memory.
* supports the creation and training of convolutional neural networks.
* only has one dependency ([FunctionalPlus](https://github.com/Dobiasd/FunctionalPlus))
* can use multiple CPUs.
* is quite slow nonetheless.
* does not make use of GPUs.
* is a needlessly reinvented wheel.
* possibly has some unknown bugs.
* is poorly documented.
* should probably not be used for anything serious.
* is merely intended to be a personal learning project.




todo
----

upconv fertig machen

Im Prinzip will man ja nur, dass der nächste Layer fuer andere klassen möglichst unique ist und für gleiche möglichst gleich. Man muss also eigentlich nicht das Optimum einer bestimmten Fehlerfunktion finden, sondern das Optimum von einer von sehr vielen möglichen Fehlerfunktionen.

oder jeden layer als einzelnes netzwerk autoencoder-maessig trainieren. Wenn der erste fertig ist, den zweiten als autoencoder fuer dessen output trainieren usw.

allow different strides for conv and for pool, stride 2 in conv can make pooling superfluous

eigene learning rate (momentum) für jeden layer oder sogar jedes weight? Muss die in einem conv layer nicht eh tiefer sein, weil ein weight direkt viele outputs hochdrückt? Müsste die learning rate eines layers nicht proportional zur stddev sein, die man bei random init dieses layers benutzt?

mem optimization: pool layer does not need to save last_input_, but max_pool needs markers

regularization to prevent overfitting (penalize squared weight sum, dropout. not needed for autoencoder?) jedes weight sollte nah an seiner standard-abweichung vom init sein

reduce RAM usage have training-data in mem as uchars, not as doubles

learning momentum in batch mode (statt meinem momentan mal SGD+Nesterov momentum or Adam probieren)
http://cs231n.github.io/neural-networks-3/

tests in separate cpp

Increase minibatch size during training

Mini batches: Shuffle input, same amount of every class in batch if possible

flatten nestes multilayer nets? Ideally directly create them that way

typedefs.h nach config.h umbenennen und globals da rein

clean up training.h

spiral dataset as a test ( http://cs231n.github.io/neural-networks-case-study/ ), with image output to see fit

backprop: use bfgs etc. on param-derivs

use cmake with google tests (see fplus)
convolution: different paddings, different steps. With step 2 one could get rid of pooling layers
dilated convolutions? (http://cs231n.github.io/convolutional-networks/)

layers to implement:
transposed convolution layer aka deconv aka unconv aka fractional step conv
loss layers additionally to sigmoid cross entropy?: Multiclass Support Vector Machine loss (SVM) manhattan (l1)?, euclidian (l2)?

mini-batches (multiple images concatted in one pass instead of one after another)

split a validation set from the training set to optimize the hyperparameters

rule of thumb: 10 times more training examples than parameters

evolutionary optimization instread of backprop? (momentum mutations? recombination?)
particle swarm optimization? verteilen, gute habe gravitation, andere fliegen dahin, momentum, random speed changes annealing, best reproduces with mutation and worst dies

json rausschreiben zwischendurch (losses, weight distribution) (temp, rename), html viewer dazu
visualize layer and filters as images
start one async task per filter in layer? (fplus::execute_paralelly?)

Affine layer- flow layer?
or instead of affine: http://torch.ch/blog/2015/09/07/spatial_transformers.html
Local Response Normalization layer?

Skip connection aka computational graph?
paralell_layer (nimmt split_layer-Ableitung und merge_layer-Ableitung (avg oder concat_z) und zwei listen aus layern, die auch leer sein können). Die werden dann nebeneinander ausgeführt (nicht zeitlich). skip_layer ist dann meta-Funktion, die einem was baut.

Caffee import?




### ideas/goals

image autoencoder

image classification

neural style (vielleicht mit perceptual loss: https://arxiv.org/abs/1603.08155)
Vielleicht kann man aber auch Style-Bild und Content-Bild gleichzeitig in ein Netz drücken wenn man sie einfach in der depth-Dimension hintereinander legt.

semantic morphing (e.g. faces)

video autoencoder? (Spatio-temporal? would mean matrix4d)

Video compression
zweites video dabei, was die differenzframes drin hat
anfang vom neuronalen netz koennte der codec sein und nur der FC-Layer waere das eigentliche Video
oder low-bitrate-video mit "Compression Artifacts Reduction by a Deep Convolutional Network" (https://arxiv.org/pdf/1504.06993.pdf) nachverbessern?