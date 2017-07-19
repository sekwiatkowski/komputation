package shape.komputation.layers.forward

import jcuda.jcublas.cublasHandle
import shape.komputation.cpu.layers.forward.CpuDenseLayer
import shape.komputation.cpu.layers.forward.activation.cpuActivationLayer
import shape.komputation.cuda.CudaContext
import shape.komputation.cuda.layers.CudaForwardLayer
import shape.komputation.cuda.layers.forward.activation.cudaActivationLayer
import shape.komputation.cuda.layers.forward.dense.CudaDenseLayer
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.CudaForwardLayerInstruction
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.forward.activation.ActivationFunction
import shape.komputation.layers.forward.projection.projectionLayer
import shape.komputation.optimization.OptimizationInstruction

class DenseLayer(
    private val name : String?,
    private val inputDimension : Int,
    private val outputDimension : Int,
    private val weightInitialization: InitializationStrategy,
    private val biasInitialization: InitializationStrategy?,
    private val activationFunction: ActivationFunction,
    private val optimization: OptimizationInstruction?) : CpuForwardLayerInstruction, CudaForwardLayerInstruction {

    private val projectionName = concatenateNames(this.name, "projection")
    private val activationName = concatenateNames(this.name, "activation")

    private val projection = projectionLayer(this.projectionName, this.inputDimension, this.outputDimension, this.weightInitialization, this.biasInitialization, this.optimization)

    override fun buildForCpu(): CpuDenseLayer {

        val projectionLayer = projection.buildForCpu()

        val activationLayer = cpuActivationLayer(this.activationName, this.activationFunction, this.outputDimension, 1).buildForCpu()

        return CpuDenseLayer(this.name, projectionLayer, activationLayer)

    }

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle): CudaForwardLayer {

        val projectionLayer = projection.buildForCuda(context, cublasHandle)

        val activationLayer = cudaActivationLayer(this.activationName, this.activationFunction, this.outputDimension).buildForCuda(context, cublasHandle)

        return CudaDenseLayer(this.name, projectionLayer, activationLayer)

    }


}

fun denseLayer(
    inputDimension : Int,
    outputDimension : Int,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    activationFunction: ActivationFunction,
    optimizationStrategy: OptimizationInstruction?) =

    denseLayer(
        null,
        inputDimension,
        outputDimension,
        weightInitializationStrategy,
        biasInitializationStrategy,
        activationFunction,
        optimizationStrategy
    )

fun denseLayer(
    name : String?,
    inputDimension : Int,
    outputDimension : Int,
    weightInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy?,
    activationFunction: ActivationFunction,
    optimization: OptimizationInstruction?) =

    DenseLayer(
        name,
        inputDimension,
        outputDimension,
        weightInitialization,
        biasInitialization,
        activationFunction,
        optimization)