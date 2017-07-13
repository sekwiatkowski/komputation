package shape.komputation.layers.forward.convolution

import shape.komputation.cpu.layers.forward.convolution.CpuMaxPoolingLayer
import shape.komputation.layers.CpuForwardLayerInstruction

class MaxPoolingLayer(private val name : String?) : CpuForwardLayerInstruction {

    override fun buildForCpu() =

        CpuMaxPoolingLayer(this.name)

}


fun maxPoolingLayer(name : String? = null) = MaxPoolingLayer(name)