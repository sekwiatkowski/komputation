package shape.komputation.layers.forward.activation

import shape.komputation.cpu.forward.activation.TempCudaSigmoidLayer
import shape.komputation.cuda.CudaEnvironment
import shape.komputation.layers.CpuForwardLayerInstruction

class CudaSigmoidLayer(
    private val name : String? = null,
    private val environment: CudaEnvironment,
    private val inputDimension : Int) : CpuForwardLayerInstruction {

    override fun buildForCpu() =

        TempCudaSigmoidLayer(this.name, this.environment.computeCapabilities, this.environment.numberThreadsPerBlock, this.inputDimension)

}

fun cudaSigmoidLayer(
    environment: CudaEnvironment,
    inputDimension : Int) =

    cudaSigmoidLayer(null, environment, inputDimension)


fun cudaSigmoidLayer(
    name : String? = null,
    environment: CudaEnvironment,
    inputDimension : Int) =

    CudaSigmoidLayer(name, environment, inputDimension)