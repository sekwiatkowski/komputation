<img src="Logo.jpg" align="right" height="150" width="150" />

# Komputation

Komputation is a neural network framework for the JVM written in the Kotlin programming language.

## Layers

- Entry points:
  - [Input](./src/main/kotlin/shape/komputation/layers/entry/InputLayer.kt)
  - [Lookup](./src/main/kotlin/shape/komputation/layers/entry/LookupLayer.kt)

- Standard feed-forward networks:
  - [Projection](./src/main/kotlin/shape/komputation/layers/feedforward/projection/ProjectionLayer.kt)

- Convolutional neural networks:
  - [Convolution](./src/main/kotlin/shape/komputation/layers/feedforward/convolution/ConvolutionalLayer.kt)
  - [Max-pooling](./src/main/kotlin/shape/komputation/layers/feedforward/convolution/MaxPoolingLayer.kt)

- Recurrent neural networks:
  - Encoder
    - [single output](./src/main/kotlin/shape/komputation/layers/feedforward/encoder/SingleOutputEncoder.kt)
    - [multi-output](./src/main/kotlin/shape/komputation/layers/feedforward/encoder/MultiOutputEncoder.kt)
  - Decoder
    - [single input](./src/main/kotlin/shape/komputation/layers/feedforward/decoder/SingleInputDecoder.kt)
    - [multi-input](./src/main/kotlin/shape/komputation/layers/feedforward/decoder/MultiInputDecoder.kt)
    - [attentive](./src/main/kotlin/shape/komputation/layers/feedforward/decoder/AttentiveDecoder.kt)

- Activation functions:
  - [Identity](./src/main/kotlin/shape/komputation/layers/feedforward/IdentityLayer.kt)
  - [Rectified Linear Units (ReLUs)](./src/main/kotlin/shape/komputation/layers/feedforward/activation/ReluLayer.kt)
  - [Sigmoid](./src/main/kotlin/shape/komputation/layers/feedforward/activation/SigmoidLayer.kt)
  - Softmax:
    - [column-wise](./src/main/kotlin/shape/komputation/layers/feedforward/activation/SoftmaxLayer.kt)
    - [vectorial](./src/main/kotlin/shape/komputation/layers/feedforward/activation/SoftmaxVectorLayer.kt)
  - [Tanh](./src/main/kotlin/shape/komputation/layers/feedforward/activation/TanhLayer.kt)

- Other layers:
  - [Transposition](./src/main/kotlin/shape/komputation/layers/feedforward/TranspositionLayer.kt)
  - [Column repetition](./src/main/kotlin/shape/komputation/layers/feedforward/ColumnRepetitionLayer.kt)

## Demos

- Boolean functions:
  - [AND](./src/main/kotlin/shape/komputation/demos/and/AndSigmoid.kt)
  - [Negation](./src/main/kotlin/shape/komputation/demos/negation/Negation.kt)
  - [XOR](./src/main/kotlin/shape/komputation/demos/xor/Xor.kt)

- Sequential data:
  - [Running total](./src/main/kotlin/shape/komputation/demos/runningtotal/RunningTotalMultiInputDecoder.kt)
  - [Addition problem](./src/main/kotlin/shape/komputation/demos/addition/AdditionProblem.kt)
  - [Reverse](./src/main/kotlin/shape/komputation/demos/reverse/ReverseSingleEncoding.kt)
  - [Reverse with attention](./src/main/kotlin/shape/komputation/demos/reverse/ReverseAttention.kt)

- Toy problems:
  - [Image classification](./src/main/kotlin/shape/komputation/demos/lines/Lines.kt)
  - [Word embeddings](./src/main/kotlin/shape/komputation/demos/embeddings/Embeddings.kt)

- NLP:
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