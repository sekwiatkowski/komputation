<img src="Logo.jpg" align="right" height="150" width="150" />

# Komputation

Komputation is a neural network framework for the JVM written in the Kotlin programming language.

## Layers

- Entry points:
  - [Input](./src/main/kotlin/shape/komputation/layers/entry/InputLayer.kt)
  - [Lookup](./src/main/kotlin/shape/komputation/layers/entry/LookupLayer.kt)

- Standard feed-forward networks:
  - [Projection](./src/main/kotlin/shape/komputation/layers/forward/projection/ProjectionLayer.kt)
  - [Dense](./src/main/kotlin/shape/komputation/layers/forward/DenseLayer.kt)

- Convolutional neural networks (CNNs):
  - [Convolution](./src/main/kotlin/shape/komputation/layers/forward/convolution/ConvolutionalLayer.kt)
  - [Max-pooling](./src/main/kotlin/shape/komputation/layers/forward/convolution/MaxPoolingLayer.kt)

- Recurrent neural networks (RNNs):
  - Encoder
    - [single output](./src/main/kotlin/shape/komputation/layers/forward/encoder/SingleOutputEncoder.kt)
    - [multi-output](./src/main/kotlin/shape/komputation/layers/forward/encoder/MultiOutputEncoder.kt)
  - Decoder
    - [single input](./src/main/kotlin/shape/komputation/layers/forward/decoder/SingleInputDecoder.kt)
    - [multi-input](./src/main/kotlin/shape/komputation/layers/forward/decoder/MultiInputDecoder.kt)
    - [attentive](./src/main/kotlin/shape/komputation/layers/forward/decoder/AttentiveDecoder.kt)

- RNN units:
  - [Simple recurrent unit](./src/main/kotlin/shape/komputation/layers/forward/units/SimpleRecurrentUnit.kt)
  - [Minimal Gated Unit](./src/main/kotlin/shape/komputation/layers/forward/units/MinimalGatedUnit.kt)

- [Dropout](./src/main/kotlin/shape/komputation/layers/forward/dropout/DropoutLayer.kt)

- [Highway layer](./src/main/kotlin/shape/komputation/layers/forward/HighwayLayer.kt)

- Activation functions:
  - [Identity](./src/main/kotlin/shape/komputation/layers/forward/IdentityLayer.kt)
  - [Rectified Linear Units (ReLUs)](./src/main/kotlin/shape/komputation/layers/forward/activation/ReluLayer.kt)
  - [Sigmoid](./src/main/kotlin/shape/komputation/layers/forward/activation/SigmoidLayer.kt)
  - Softmax:
    - [column-wise](./src/main/kotlin/shape/komputation/layers/forward/activation/SoftmaxLayer.kt)
    - [vectorial](./src/main/kotlin/shape/komputation/layers/forward/activation/SoftmaxVectorLayer.kt)
  - [Tanh](./src/main/kotlin/shape/komputation/layers/forward/activation/TanhLayer.kt)

- Other layers:
  - [Concatenation](./src/main/kotlin/shape/komputation/layers/forward/Concatenation.kt)
  - [Transposition](./src/main/kotlin/shape/komputation/layers/forward/TranspositionLayer.kt)
  - [Counter-probability](./src/main/kotlin/shape/komputation/layers/forward/CounterProbabilityLayer.kt)
  - [Column repetition](./src/main/kotlin/shape/komputation/layers/forward/ColumnRepetitionLayer.kt)

## Demos

- Boolean functions:
  - [AND](./src/main/kotlin/shape/komputation/demos/and/AndSigmoid.kt)
  - [Negation](./src/main/kotlin/shape/komputation/demos/negation/Negation.kt)
  - [XOR](./src/main/kotlin/shape/komputation/demos/xor/Xor.kt)

- Running total:
  - [Projection](./src/main/kotlin/shape/komputation/demos/runningtotal/RunningTotalProjection.kt)
  - [Multi-input decoder](./src/main/kotlin/shape/komputation/demos/runningtotal/RunningTotalMultiInputDecoder.kt)

- [Computer vision toy problem](./src/main/kotlin/shape/komputation/demos/lines/Lines.kt)

- Word embedding toy problem:
  - [CNN with one filter height](./src/main/kotlin/shape/komputation/demos/embeddings/Embeddings.kt)
  - [CNN with two filter heights](./src/main/kotlin/shape/komputation/demos/embeddings/EmbeddingsWithDifferentFilterHeights.kt)

- Addition problem:
  - [Simple recurrent unit](./src/main/kotlin/shape/komputation/demos/addition/AdditionProblemRecurrentUnit.kt)
  - [Minimal Gated Unit](./src/main/kotlin/shape/komputation/demos/addition/AdditionProblemMGU.kt)

- Reverse function:
  - [Unidirectional RNN](./src/main/kotlin/shape/komputation/demos/reverse/ReverseUnidirectional.kt)
  - [Bidirectional RNN](./src/main/kotlin/shape/komputation/demos/reverse/ReverseBidirectional.kt)
  - [RNN with attention](./src/main/kotlin/shape/komputation/demos/reverse/ReverseAttention.kt)

- MNIST:
  - [Dropout](./src/main/kotlin/shape/komputation/demos/mnist/MnistDropout.kt)
  - [He initialization, dropout and Nesterov's Accelerated Gradient](./src/main/kotlin/shape/komputation/demos/mnist/MnistHeDropoutNesterov.kt)
  - [Highway layers](./src/main/kotlin/shape/komputation/demos/mnist/MnistHighway.kt)

- [TREC question classification](./src/main/kotlin/shape/komputation/demos/trec/TREC.kt)

## Sample code

The following code instantiates a convolutional neural network for sentence classification:

 ```kotlin
val network = Network(
    lookupLayer(embeddings, embeddingDimension, maximumBatchSize, optimizationStrategy),
    concatenation(
        *filterHeights
            .map { filterHeight ->
                arrayOf(
                    convolutionalLayer(numberFilters, filterWidth, filterHeight, initializationStrategy, optimizationStrategy),
                    reluLayer(),
                    maxPoolingLayer()
                )
            }
            .toTypedArray()
    ),
    projectionLayer(numberFilters * numberFilterHeights, numberCategories, initializationStrategy, initializationStrategy, optimizationStrategy),
    softmaxLayer()
)
```

See the [TREC demo](./src/main/kotlin/shape/komputation/demos/trec/TREC.kt) for more details.

## Initialization

- [Constant](./src/main/kotlin/shape/komputation/initialization/ConstantInitialization.kt)
- [Gaussian](./src/main/kotlin/shape/komputation/initialization/GaussianInitialization.kt)
- [He](./src/main/kotlin/shape/komputation/initialization/HeInitialization.kt)
- [Identity](./src/main/kotlin/shape/komputation/initialization/IdentityInitialization.kt)
- [Uniform](./src/main/kotlin/shape/komputation/initialization/UniformInitialization.kt)
- [Zero](./src/main/kotlin/shape/komputation/initialization/ZeroInitialization.kt)

## Loss functions

- [Logistic loss](./src/main/kotlin/shape/komputation/loss/LogisticLoss.kt)
- [Squared loss](./src/main/kotlin/shape/komputation/loss/SquaredLoss.kt)

## Optimization

- [Stochastic Gradient Descent](./src/main/kotlin/shape/komputation/optimization/StochasticGradientDescent.kt)
- Momentum-based:
  - [Momentum](./src/main/kotlin/shape/komputation/optimization/Momentum.kt)
  - [Nesterov's Accelerated Gradient](./src/main/kotlin/shape/komputation/optimization/Nesterov.kt)
- Adaptive learning rates:
  - [Adagrad](./src/main/kotlin/shape/komputation/optimization/Adagrad.kt)
  - [Adadelta](./src/main/kotlin/shape/komputation/optimization/Adadelta.kt)
  - [RMSProp](./src/main/kotlin/shape/komputation/optimization/RMSProp.kt)