package shape.komputation.cuda.layers.forward.activation

import shape.komputation.cuda.layers.CudaForwardLayer
import shape.komputation.layers.CudaActivationLayerInstruction
import shape.komputation.layers.forward.activation.ActivationFunction
import shape.komputation.layers.forward.activation.sigmoidLayer

fun cudaActivationLayer(name: String?, function: ActivationFunction, dimension : Int) : CudaActivationLayerInstruction =

    when (function) {

        ActivationFunction.Sigmoid ->
            sigmoidLayer(name, dimension)
        else ->
            throw NotImplementedError()

    }


interface CudaActivationLayer : CudaForwardLayer