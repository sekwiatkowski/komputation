package shape.komputation.cuda.layers.forward.activation

import shape.komputation.cuda.layers.CudaForwardLayer
import shape.komputation.layers.CudaActivationLayerInstruction
import shape.komputation.layers.forward.activation.ActivationFunction
import shape.komputation.layers.forward.activation.reluLayer
import shape.komputation.layers.forward.activation.sigmoidLayer
import shape.komputation.layers.forward.activation.softmaxLayer

fun cudaActivationLayer(name: String?, function: ActivationFunction, dimension : Int) : CudaActivationLayerInstruction =

    when (function) {

        ActivationFunction.Sigmoid ->
            sigmoidLayer(name, dimension)
        ActivationFunction.ReLU ->
            reluLayer(name, dimension)
        ActivationFunction.Softmax ->
            softmaxLayer(name, dimension)
        else ->
            throw NotImplementedError()

    }


interface CudaActivationLayer : CudaForwardLayer