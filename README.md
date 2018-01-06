<img src="Logo.jpg" align="right" height="150" width="150" />

# Komputation

Komputation is a neural network framework for the Java Virtual Machine written in the Kotlin programming language.

## Maven

Komputation is available through Maven Central:

```xml
<dependency>
    <groupId>com.komputation</groupId>
    <artifactId>komputation</artifactId>
    <version>0.12.1</version>
</dependency>
```

## Layers

- Entry points:
  - [Input](./src/main/kotlin/com/komputation/instructions/entry/Input.kt)
  - [Lookup](./src/main/kotlin/com/komputation/instructions/entry/Lookup.kt)

- Standard feed-forward networks:
  - [Weighting](./src/main/kotlin/com/komputation/instructions/continuation/projection/Weighting.kt)
  - [Bias](./src/main/kotlin/com/komputation/instructions/continuation/projection/Bias.kt)
  - [Projection](./src/main/kotlin/com/komputation/instructions/continuation/projection/Projection.kt)
  - [Dense](./src/main/kotlin/com/komputation/instructions/continuation/dense/Dense.kt)

- Convolutional neural networks (CNNs):
  - [Convolution](./src/main/kotlin/com/komputation/instructions/continuation/convolution/Convolution.kt)
  - [Max-pooling](./src/main/kotlin/com/komputation/instructions/continuation/convolution/MaxPooling.kt)

- Recurrent neural networks:
  - [Recurrent layer](./src/main/kotlin/com/komputation/instructions/recurrent/Recurrent.kt)
  - [Bidirectional recurrent layer](./src/main/kotlin/com/komputation/instructions/recurrent/BidirectionalRecurrent.kt)

- [Dropout](./src/main/kotlin/com/komputation/instructions/continuation/dropout/Dropout.kt)

- Activation functions:
  - [Identity](./src/main/kotlin/com/komputation/instructions/continuation/activation/Identity.kt)
  - [Rectified Linear Units (ReLUs)](./src/main/kotlin/com/komputation/instructions/continuation/activation/Relu.kt)
  - [Sigmoid](./src/main/kotlin/com/komputation/instructions/continuation/activation/Sigmoid.kt)
  - [Softmax](./src/main/kotlin/com/komputation/instructions/continuation/activation/Softmax.kt)
  - [Tanh](./src/main/kotlin/com/komputation/instructions/continuation/activation/Tanh.kt)

- Other layers:
  - [Stack](./src/main/kotlin/com/komputation/instructions/continuation/stack/stack.kt)
  - [Exponentiation](./src/main/kotlin/com/komputation/instructions/continuation/activation/ExponentiationLayer.kt)
  - [Normalization](./src/main/kotlin/com/komputation/instructions/continuation/NormalizationLayer.kt)

## CPU demos

- Boolean functions:
  - [AND](./src/main/kotlin/com/komputation/cpu/demos/and/AndSigmoid.kt)
  - [NOT](./src/main/kotlin/com/komputation/cpu/demos/not/Not.kt)
  - [XOR](./src/main/kotlin/com/komputation/cpu/demos/xor/Xor.kt)

- Total:
  - [Fixed length](./src/main/kotlin/com/komputation/cpu/demos/total/FixedLengthTotal.kt)
  - [Variable length](./src/main/kotlin/com/komputation/cpu/demos/total/VariableLengthTotal.kt)

- Running total:
  - Left-to-right:
    - [Fixed length](./src/main/kotlin/com/komputation/cpu/demos/runningtotal/lefttoright/FixedLengthRunningTotal.kt)
    - [Variable length](./src/main/kotlin/com/komputation/cpu/demos/runningtotal/lefttoright/VariableLengthRunningTotal.kt)
  - Right-to-left:
    - [Fixed length](./src/main/kotlin/com/komputation/cpu/demos/runningtotal/righttoleft/RightToLeftFixedLengthRunningTotal.kt)
    - [Variable length](./src/main/kotlin/com/komputation/cpu/demos/runningtotal/righttoleft/RightToLeftVariableLengthRunningTotal.kt)
  - Bidirectional:
    - [Fixed length](./src/main/kotlin/com/komputation/cpu/demos/runningtotal/bidirectional/BidirectionalFixedLengthRunningTotal.kt)
    - [Variable length](./src/main/kotlin/com/komputation/cpu/demos/runningtotal/bidirectional/BidirectionalVariableLengthRunningTotal.kt)

- Increment:
  - [One layer](./src/main/kotlin/com/komputation/cpu/demos/increment/Increment.kt)
  - [Two layers](./src/main/kotlin/com/komputation/cpu/demos/increment/IncrementTwice.kt)

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
  - [CNN with two filter widths](./src/main/kotlin/com/komputation/cuda/demos/embeddings/EmbeddingsWithTwoFilterWidths.kt)

- MNIST:
  - [Minimal](./src/main/kotlin/com/komputation/cuda/demos/mnist/MnistMinimal.kt)
  - [Dropout](./src/main/kotlin/com/komputation/cuda/demos/mnist/MnistBatchDropout.kt)

- TREC:
  - [One filter width](./src/main/kotlin/com/komputation/cuda/demos/trec/TREC.kt)
  - [Two filter widths](./src/main/kotlin/com/komputation/cuda/demos/trec/TRECWithTwoFilterWidths.kt)

## Sample code

The following code instantiates a GPU-accelerated convolutional neural network for sentence classification:

```kotlin
    val sentenceClassifier = cudaNetwork(
        batchSize,
        lookup(embeddings, maximumDocumentLength, embeddingDimension, optimization),
        convolution(numberFilters, filterWidth, filterHeight, initialization, optimization),
        relu(),
        dropout(random, keepProbability),
        dense(numberCategories, Activation.Softmax, initialization, optimization)
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

- [Cross-entropy loss](./src/main/kotlin/com/komputation/instructions/loss/CrossEntropyLoss.kt)
- [Logistic loss](./src/main/kotlin/com/komputation/instructions/loss/LogisticLoss.kt)
- [Squared loss](./src/main/kotlin/com/komputation/instructions/loss/SquaredLoss.kt)

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