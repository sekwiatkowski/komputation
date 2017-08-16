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
  - [Simple recurrent unit](./src/main/kotlin/shape/komputation/cpu/layers/forward/units/SimpleRecurrentUnit.kt)
  - [Minimal Gated Unit](./src/main/kotlin/shape/komputation/cpu/layers/forward/units/MinimalGatedUnit.kt)

- [Dropout](./src/main/kotlin/shape/komputation/layers/forward/dropout/DropoutLayer.kt)

- [Highway layer](./src/main/kotlin/shape/komputation/layers/forward/HighwayLayer.kt)

- Activation functions:
  - [Identity](./src/main/kotlin/shape/komputation/layers/forward/activation/IdentityLayer.kt)
  - [Rectified Linear Units (ReLUs)](./src/main/kotlin/shape/komputation/layers/forward/activation/ReluLayer.kt)
  - [Sigmoid](./src/main/kotlin/shape/komputation/layers/forward/activation/SigmoidLayer.kt)
  - [Softmax](./src/main/kotlin/shape/komputation/layers/forward/activation/SoftmaxLayer.kt)
  - [Tanh](./src/main/kotlin/shape/komputation/layers/forward/activation/TanhLayer.kt)

- Other layers:
  - [Column repetition](./src/main/kotlin/shape/komputation/layers/forward/ColumnRepetitionLayer.kt)
  - [Concatenation](./src/main/kotlin/shape/komputation/layers/forward/Concatenation.kt)
  - [Counter-probability](./src/main/kotlin/shape/komputation/layers/forward/CounterProbabilityLayer.kt)
  - [Exponentiation](./src/main/kotlin/shape/komputation/layers/forward/activation/ExponentiationLayer.kt)
  - [Normalization](./src/main/kotlin/shape/komputation/layers/forward/NormalizationLayer.kt)
  - [Transposition](./src/main/kotlin/shape/komputation/layers/forward/TranspositionLayer.kt)

## Demos

- Boolean functions:
  - [AND](./src/main/kotlin/shape/komputation/cpu/demos/and/AndSigmoid.kt)
  - [Negation](./src/main/kotlin/shape/komputation/cpu/demos/negation/Negation.kt)
  - [XOR](./src/main/kotlin/shape/komputation/cpu/demos/xor/Xor.kt)

- Running total:
  - [Projection](./src/main/kotlin/shape/komputation/cpu/demos/runningtotal/RunningTotalProjection.kt)
  - [Multi-input decoder](./src/main/kotlin/shape/komputation/cpu/demos/runningtotal/RunningTotalMultiInputDecoder.kt)

- [Computer vision toy problem](./src/main/kotlin/shape/komputation/cpu/demos/lines/Lines.kt)

- Word embedding toy problem:
  - [CNN with one filter width](./src/main/kotlin/shape/komputation/cpu/demos/embeddings/Embeddings.kt)
  - [CNN with two filter widths](./src/main/kotlin/shape/komputation/cpu/demos/embeddings/EmbeddingsWithDifferentFilterHeights.kt)

- Addition problem:
  - [Simple recurrent unit](./src/main/kotlin/shape/komputation/cpu/demos/addition/AdditionProblemRecurrentUnit.kt)
  - [Minimal Gated Unit](./src/main/kotlin/shape/komputation/cpu/demos/addition/AdditionProblemMGU.kt)

- Reverse function:
  - [Unidirectional RNN](./src/main/kotlin/shape/komputation/cpu/demos/reverse/ReverseUnidirectional.kt)
  - [Bidirectional RNN](./src/main/kotlin/shape/komputation/cpu/demos/reverse/ReverseBidirectional.kt)
  - [RNN with attention](./src/main/kotlin/shape/komputation/cpu/demos/reverse/ReverseAttention.kt)

- MNIST:
  - [CPU demo](./src/main/kotlin/shape/komputation/cpu/demos/mnist/MnistBatchDropout.kt)
  - [GPU/CUDA demo](./src/main/kotlin/shape/komputation/cuda/demos/mnist/MnistBatchDropout.kt)

- [TREC question classification](./src/main/kotlin/shape/komputation/cpu/demos/trec/TREC.kt)

## Sample code

The following code instantiates a convolutional neural network for sentence classification:

 ```kotlin
    val network = Network(
        batchSize,
        lookupLayer(embeddings, maximumDocumentLength, hasFixedLength, embeddingDimension, optimization),
        concatenation(
            embeddingDimension,
            maximumDocumentLength,
            false,
            IntArray(numberFilterWidths) { numberFilters },
            1,
            filterWidths
                .map { filterWidth ->
                    convolutionalLayer(embeddingDimension, maximumDocumentLength, hasFixedLength, numberFilters, filterWidth, filterHeight, initialization, initialization, optimization)
                }
                .toTypedArray()
        ),
        reluLayer(numberFilterWidths * numberFilters),
        dropoutLayer(random, keepProbability, numberFilterWidths * numberFilters),
        projectionLayer(numberFilterWidths * numberFilters, numberCategories, initialization, initialization, optimization),
        softmaxLayer(numberCategories)
    )
```

See the [TREC demo](./src/main/kotlin/shape/komputation/cpus/demos/trec/TREC.kt) for more details.

## Initialization

- [Provided](./src/main/kotlin/shape/komputation/initialization/ProvidedInitialization.kt)
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
- Historical:
  - [Momentum](./src/main/kotlin/shape/komputation/optimization/historical/Momentum.kt)
  - [Nesterov's Accelerated Gradient](./src/main/kotlin/shape/komputation/optimization/historical/Nesterov.kt)
- Adaptive:
  - [Adagrad](./src/main/kotlin/shape/komputation/optimization/adaptive/Adagrad.kt)
  - [Adadelta](./src/main/kotlin/shape/komputation/optimization/adaptive/Adadelta.kt)
  - [RMSProp](./src/main/kotlin/shape/komputation/optimization/adaptive/RMSProp.kt)
  - [Adam](./src/main/kotlin/shape/komputation/optimization/adaptive/Adam.kt)