<img src="Logo.jpg" align="right" height="150" width="150" />

# Komputation

Komputation is a neural network framework for the JVM written in the Kotlin programming language.

## Maven

Komputation is available through Maven Central:

```xml
<dependency>
    <groupId>com.komputation</groupId>
    <artifactId>komputation</artifactId>
    <version>0.11.2</version>
</dependency>
```

## Layers

- Entry points:
  - [Input](./src/main/kotlin/com/komputation/layers/entry/InputLayer.kt)
  - [Lookup](./src/main/kotlin/com/komputation/layers/entry/LookupLayer.kt)

- Standard feed-forward networks:
  - [Projection](./src/main/kotlin/com/komputation/layers/forward/projection/ProjectionLayer.kt)
  - [Dense](./src/main/kotlin/com/komputation/layers/forward/dense/DenseLayer.kt)

- Convolutional neural networks (CNNs):
  - [Convolution](./src/main/kotlin/com/komputation/layers/forward/convolution/ConvolutionLayer.kt)
  - [Max-pooling](./src/main/kotlin/com/komputation/layers/forward/convolution/MaxPoolingLayer.kt)

- Recurrent neural networks:
  - [Recurrent layer](./src/main/kotlin/com/komputation/layers/recurrent/RecurrentLayer.kt)
    - [All steps](./src/main/kotlin/com/komputation/cpu/layers/recurrent/extraction/AllSteps.kt)
    - [Last step](./src/main/kotlin/com/komputation/cpu/layers/recurrent/extraction/LastStep.kt)

- [Dropout](./src/main/kotlin/com/komputation/layers/forward/dropout/DropoutLayer.kt)

- Activation functions:
  - [Identity](./src/main/kotlin/com/komputation/layers/forward/activation/IdentityLayer.kt)
  - [Rectified Linear Units (ReLUs)](./src/main/kotlin/com/komputation/layers/forward/activation/ReluLayer.kt)
  - [Sigmoid](./src/main/kotlin/com/komputation/layers/forward/activation/SigmoidLayer.kt)
  - [Softmax](./src/main/kotlin/com/komputation/layers/forward/activation/SoftmaxLayer.kt)
  - [Tanh](./src/main/kotlin/com/komputation/layers/forward/activation/TanhLayer.kt)

- Other layers:
  - [Concatenation](./src/main/kotlin/com/komputation/layers/forward/Concatenation.kt)
  - [Exponentiation](./src/main/kotlin/com/komputation/layers/forward/activation/ExponentiationLayer.kt)
  - [Normalization](./src/main/kotlin/com/komputation/layers/forward/NormalizationLayer.kt)

## CPU demos

- Boolean functions:
  - [AND](./src/main/kotlin/com/komputation/cpu/demos/and/AndSigmoid.kt)
  - [Negation](./src/main/kotlin/com/komputation/cpu/demos/negation/Negation.kt)
  - [XOR](./src/main/kotlin/com/komputation/cpu/demos/xor/Xor.kt)

- Total:
  - [Fixed length](./src/main/kotlin/com/komputation/cpu/demos/total/FixedLengthRunningTotal.kt)
  - [Variable length](./src/main/kotlin/com/komputation/cpu/demos/total/VariableLengthRunningTotal.kt)

- Running total:
  - Forward:
    - [Fixed length](./src/main/kotlin/com/komputation/cpu/demos/runningtotal/forward/FixedLengthRunningTotal.kt)
    - [Variable length](./src/main/kotlin/com/komputation/cpu/demos/runningtotal/forward/VariableLengthRunningTotal.kt)
  - Backward:
    - [Fixed length](./src/main/kotlin/com/komputation/cpu/demos/runningtotal/backward/BackwardFixedLengthRunningTotal.kt)
    - [Variable length](./src/main/kotlin/com/komputation/cpu/demos/runningtotal/backward/BackwardVariableLengthRunningTotal.kt)

- [Increment](./src/main/kotlin/com/komputation/cpu/demos/increment/Increment.kt)

- Word embedding toy problem:
  - [Feed-forward network](./src/main/kotlin/com/komputation/cpu/demos/embeddings/Embeddings.kt)
  - [CNN with one filter width](./src/main/kotlin/com/komputation/cpu/demos/embeddings/EmbeddingsWithConvolution.kt)
  - [CNN with two filter widths](./src/main/kotlin/com/komputation/cpu/demos/embeddings/EmbeddingsWithTwoFilterWidths.kt)

- [Sequence labeling toy problem](./src/main/kotlin/com/komputation/cpu/demos/sequencelabeling/SequenceLabeling.kt)

- [Computer vision toy problem](./src/main/kotlin/com/komputation/cpu/demos/lines/Lines.kt)

- MNIST:
  - [Minimal](./src/main/kotlin/com/komputation/cpu/demos/mnist/MnistMinimal.kt)
  - [Dropout](./src/main/kotlin/com/komputation/cpu/demos/mnist/MnistBatchDropout.kt)

- TREC:
  - [One filter width](./src/main/kotlin/com/komputation/cpu/demos/trec/TREC.kt)
  - [Two filter widths](./src/main/kotlin/com/komputation/cpu/demos/trec/TRECWithTwoFilterWidths.kt)

## GPU/CUDA demos

- Boolean functions:
  - [AND](./src/main/kotlin/com/komputation/cuda/demos/and/AndSigmoid.kt)
  - [Negation](./src/main/kotlin/com/komputation/cuda/demos/negation/Negation.kt)
  - [XOR](./src/main/kotlin/com/komputation/cuda/demos/xor/Xor.kt)

- Word embedding toy problem:
  - [Feed-forward network](./src/main/kotlin/com/komputation/cuda/demos/embeddings/Embeddings.kt)
  - [CNN with one filter width](./src/main/kotlin/com/komputation/cuda/demos/embeddings/EmbeddingsWithConvolution.kt)

- MNIST:
  - [Minimal](./src/main/kotlin/com/komputation/cuda/demos/mnist/MnistMinimal.kt)
  - [Dropout](./src/main/kotlin/com/komputation/cuda/demos/mnist/MnistBatchDropout.kt)

- TREC:
  - [One filter width](./src/main/kotlin/com/komputation/cuda/demos/trec/TREC.kt)

## Sample code

The following code instantiates a GPU-accelerated convolutional neural network for sentence classification:

```kotlin
    val network = CudaNetwork(
        batchSize,
        lookupLayer(embeddings, maximumDocumentLength, hasFixedLength, embeddingDimension, optimization),
        convolutionalLayer(embeddingDimension, maximumDocumentLength, hasFixedLength, numberFilters, filterWidth, filterHeight, weightInitialization, biasInitialization, optimization),
        reluLayer(numberFilters),
        dropoutLayer(random, keepProbability, numberFilters),
        projectionLayer(numberFilters, numberCategories, weightInitialization, biasInitialization, optimization),
        softmaxLayer(numberCategories)
    )
```

See the [TREC demo](./src/main/kotlin/com/komputation/cuda/demos/trec/TREC.kt) for more details.

## Initialization

- [Provided](./src/main/kotlin/com/komputation/initialization/ProvidedInitialization.kt)
- [Constant](./src/main/kotlin/com/komputation/initialization/ConstantInitialization.kt)
- [Gaussian](./src/main/kotlin/com/komputation/initialization/GaussianInitialization.kt)
- [He](./src/main/kotlin/com/komputation/initialization/HeInitialization.kt)
- [Identity](./src/main/kotlin/com/komputation/initialization/IdentityInitialization.kt)
- [Uniform](./src/main/kotlin/com/komputation/initialization/UniformInitialization.kt)
- [Zero](./src/main/kotlin/com/komputation/initialization/ZeroInitialization.kt)

## Loss functions

- [Cross-entropy loss](./src/main/kotlin/com/komputation/loss/CrossEntropyLoss.kt)
- [Logistic loss](./src/main/kotlin/com/komputation/loss/LogisticLoss.kt)
- [Squared loss](./src/main/kotlin/com/komputation/loss/SquaredLoss.kt)

## Optimization

- [Stochastic Gradient Descent](./src/main/kotlin/com/komputation/optimization/StochasticGradientDescent.kt)
- Historical:
  - [Momentum](./src/main/kotlin/com/komputation/optimization/historical/Momentum.kt)
  - [Nesterov's Accelerated Gradient](./src/main/kotlin/com/komputation/optimization/historical/Nesterov.kt)
- Adaptive:
  - [Adagrad](./src/main/kotlin/com/komputation/optimization/adaptive/Adagrad.kt)
  - [Adadelta](./src/main/kotlin/com/komputation/optimization/adaptive/Adadelta.kt)
  - [RMSProp](./src/main/kotlin/com/komputation/optimization/adaptive/RMSProp.kt)
  - [Adam](./src/main/kotlin/com/komputation/optimization/adaptive/Adam.kt)