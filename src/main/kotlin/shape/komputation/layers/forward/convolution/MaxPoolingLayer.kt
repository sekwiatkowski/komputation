package shape.komputation.layers.forward.convolution

import shape.komputation.cpu.layers.forward.convolution.CpuMaxPoolingLayer
import shape.komputation.layers.CpuForwardLayerInstruction

class MaxPoolingLayer(private val name : String?, private val numberRows : Int) : CpuForwardLayerInstruction {

    override fun buildForCpu() =

        CpuMaxPoolingLayer(this.name, this.numberRows)

}


fun maxPoolingLayer(numberRows : Int) = MaxPoolingLayer(null, numberRows)

fun maxPoolingLayer(name : String? = null, numberRows : Int) = MaxPoolingLayer(name, numberRows)