<img src="Logo.jpg" align="right" height="150" width="150" />

# Komputation

Komputation is a neural network framework for the JVM written in the Kotlin programming language.

## Layers

- Entry points:
  - [Input](./src/main/kotlin/shape/komputation/layers/entry/InputLayer.kt)
  - [Lookup](./src/main/kotlin/shape/komputation/layers/entry/LookupLayer.kt)

- Standard feed-forward networks:
  - [Projection](./src/main/kotlin/shape/komputation/layers/feedforward/projection/ProjectionLayer.kt)
  - [Dense](./src/main/kotlin/shape/komputation/layers/feedforward/DenseLayer.kt)

- Convolutional neural networks (CNNs):
  - [Convolution](./src/main/kotlin/shape/komputation/layers/feedforward/convolution/ConvolutionalLayer.kt)
  - [Max-pooling](./src/main/kotlin/shape/komputation/layers/feedforward/convolution/MaxPoolingLayer.kt)

- Recurrent neural networks (RNNs):
  - Encoder
    - [single output](./src/main/kotlin/shape/komputation/layers/feedforward/encoder/SingleOutputEncoder.kt)
    - [multi-output](./src/main/kotlin/shape/komputation/layers/feedforward/encoder/MultiOutputEncoder.kt)
  - Decoder
    - [single input](./src/main/kotlin/shape/komputation/layers/feedforward/decoder/SingleInputDecoder.kt)
    - [multi-input](./src/main/kotlin/shape/komputation/layers/feedforward/decoder/MultiInputDecoder.kt)
    - [attentive](./src/main/kotlin/shape/komputation/layers/feedforward/decoder/AttentiveDecoder.kt)

- RNN units:
  - [Simple recurrent unit](./src/main/kotlin/shape/komputation/layers/feedforward/units/SimpleRecurrentUnit.kt)
  - [Minimal Gated Unit](./src/main/kotlin/shape/komputation/layers/feedforward/units/MinimalGatedUnit.kt)

- [Highway layer](./src/main/kotlin/shape/komputation/layers/feedforward/HighwayLayer.kt)

- Activation functions:
  - [Identity](./src/main/kotlin/shape/komputation/layers/feedforward/IdentityLayer.kt)
  - [Rectified Linear Units (ReLUs)](./src/main/kotlin/shape/komputation/layers/feedforward/activation/ReluLayer.kt)
  - [Sigmoid](./src/main/kotlin/shape/komputation/layers/feedforward/activation/SigmoidLayer.kt)
  - Softmax:
    - [column-wise](./src/main/kotlin/shape/komputation/layers/feedforward/activation/SoftmaxLayer.kt)
    - [vectorial](./src/main/kotlin/shape/komputation/layers/feedforward/activation/SoftmaxVectorLayer.kt)
  - [Tanh](./src/main/kotlin/shape/komputation/layers/feedforward/activation/TanhLayer.kt)

- Other layers:
  - [Concatenation](./src/main/kotlin/shape/komputation/layers/feedforward/Concatenation.kt)
  - [Transposition](./src/main/kotlin/shape/komputation/layers/feedforward/TranspositionLayer.kt)
  - [Counter-probability](./src/main/kotlin/shape/komputation/layers/feedforward/CounterProbabilityLayer.kt)
  - [Column repetition](./src/main/kotlin/shape/komputation/layers/feedforward/ColumnRepetitionLayer.kt)

## Demos

- Boolean functions:
  - [AND](./src/main/kotlin/shape/komputation/demos/and/AndSigmoid.kt)
  - [Negation](./src/main/kotlin/shape/komputation/demos/negation/Negation.kt)
  - [XOR](./src/main/kotlin/shape/komputation/demos/xor/Xor.kt)

- Addition problem:
  - [Simple recurrent unit](./src/main/kotlin/shape/komputation/demos/addition/AdditionProblemRecurrentUnit.kt)
  - [Minimal Gated Unit](./src/main/kotlin/shape/komputation/demos/addition/AdditionProblemMGU.kt)

- Reverse function:
  - [Unidirectional RNN](./src/main/kotlin/shape/komputation/demos/reverse/ReverseUnidirectional.kt)
  - [Bidirectional RNN](./src/main/kotlin/shape/komputation/demos/reverse/ReverseBidirectional.kt)
  - [RNN with attention](./src/main/kotlin/shape/komputation/demos/reverse/ReverseAttention.kt)

- Running total:
  - [Projection](./src/main/kotlin/shape/komputation/demos/runningtotal/RunningTotalProjection.kt)
  - [Multi-input decoder](./src/main/kotlin/shape/komputation/demos/runningtotal/RunningTotalMultiInputDecoder.kt)

- [Computer vision toy problem](./src/main/kotlin/shape/komputation/demos/lines/Lines.kt)

- [MNIST](./src/main/kotlin/shape/komputation/demos/mnist/Mnist.kt)

- Word embedding toy problem:
  - [CNN with one filter height](./src/main/kotlin/shape/komputation/demos/embeddings/Embeddings.kt)
  - [CNN with two filter heights](./src/main/kotlin/shape/komputation/demos/embeddings/EmbeddingsWithDifferentFilterHeights.kt)

- [TREC question classification](./src/main/kotlin/shape/komputation/demos/trec/TREC.kt)

## Sample code

The following code instantiates a convolutional neural network for sentence classification:

 ```kotlin
val network = Network(
    createLookupLayer(embeddings, optimizationStrategy),
    createConcatenation(
        *filterHeights
            .map { filterHeight ->
                arrayOf(
                    createConvolutionalLayer(numberFilters, filterWidth, filterHeight, initializationStrategy, optimizationStrategy),
                    ReluLayer(),
                    MaxPoolingLayer()
                )
            }
            .toTypedArray()
    ),
    createProjectionLayer(numberFilters * numberFilterHeights, numberCategories, initializationStrategy, optimizationStrategy),
    SoftmaxLayer()
)
```

See the [TREC demo](./src/main/kotlin/shape/komputation/demos/trec/TREC.kt) for more details.

## Initialization

- [Constant](./src/main/kotlin/shape/komputation/initialization/ConstantInitialization.kt)
- [Gaussian](./src/main/kotlin/shape/komputation/initialization/GaussianInitialization.kt)
- [Identity](./src/main/kotlin/shape/komputation/initialization/IdentityInitialization.kt)
- [Uniform](./src/main/kotlin/shape/komputation/initialization/UniformInitialization.kt)
- [Zero](./src/main/kotlin/shape/komputation/initialization/ZeroInitialization.kt)

## Loss functions

- [Logistic loss](./src/main/kotlin/shape/komputation/loss/LogisticLoss.kt)
- [Squared loss](./src/main/kotlin/shape/komputation/loss/SquaredLoss.kt)

## Optimization

- [Stochastic Gradient Descent](./src/main/kotlin/shape/komputation/optimization/StochasticGradientDescent.kt)
- [Momentum](./src/main/kotlin/shape/komputation/optimization/Momentum.kt)