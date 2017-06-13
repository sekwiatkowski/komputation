# Komputation

Komputation is a neural network framework for the JVM written in the Kotlin programming language.

## Initialization

- [Constant](./src/main/kotlin/shape/komputation/initialization/ConstantInitialization.kt)
- [Gaussian](./src/main/kotlin/shape/komputation/initialization/GaussianInitialization.kt)
- [Uniform](./src/main/kotlin/shape/komputation/initialization/UniformInitialization.kt)
- [Identity](./src/main/kotlin/shape/komputation/initialization/IdentityInitialization.kt)

## Layers

- Entry points:
  - [Input](./src/main/kotlin/shape/komputation/layers/entry/InputLayer.kt)
  - [Lookup](./src/main/kotlin/shape/komputation/layers/entry/LookupLayer.kt)

- Standard feed-forward networks:
  - [Projection](./src/main/kotlin/shape/komputation/layers/continuation/ProjectionLayer.kt)
  - [Sigmoid](./src/main/kotlin/shape/komputation/layers/continuation/SigmoidLayer.kt)
  - [Softmax](./src/main/kotlin/shape/komputation/layers/continuation/SoftmaxLayer.kt)

- Convolutional neural networks:
  - [Convolution](./src/main/kotlin/shape/komputation/layers/continuation/ConvolutionalLayer.kt)
  - [Max-pooling](./src/main/kotlin/shape/komputation/layers/continuation/MaxPoolingLayer.kt)
  - [Rectified Linear Units (ReLUs)](./src/main/kotlin/shape/komputation/layers/continuation/ReluLayer.kt)

## Loss functions

- [Logistic loss](./src/main/kotlin/shape/komputation/loss/LogisticLoss.kt)
- [Squared loss](./src/main/kotlin/shape/komputation/loss/SquaredLoss.kt)

## Optimization

- [Stochastic Gradient Descent](./src/main/kotlin/shape/komputation/optimization/StochasticGradientDescent.kt)
- [Momentum](./src/main/kotlin/shape/komputation/optimization/Momentum.kt)

## Demos

- Boolean functions:
  - [AND](./src/main/kotlin/shape/komputation/demos/AndSigmoid.kt)
  - [Negation](./src/main/kotlin/shape/komputation/demos/Negation.kt)
  - [XOR](./src/main/kotlin/shape/komputation/demos/Xor.kt)

- Toy problems:
  - [Image classification](./src/main/kotlin/shape/komputation/demos/LineDemo.kt)
  - [Word embeddings](./src/main/kotlin/shape/komputation/demos/Embeddings.kt)

- NLP:
  - [TREC question classification](./src/main/kotlin/shape/komputation/demos/TREC.kt)

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

See the [TREC demo](./src/main/kotlin/shape/komputation/demos/TREC.kt) for more details.