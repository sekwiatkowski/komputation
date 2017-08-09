package shape.komputation.layers.forward.convolution

import shape.komputation.cpu.layers.forward.convolution.CpuMaxPoolingLayer
import shape.komputation.layers.CpuForwardLayerInstruction

class MaxPoolingLayer(private val name : String?, private val numberRows : Int, private val minimumColumns : Int, private val maximumColumns : Int) : CpuForwardLayerInstruction {

    override fun buildForCpu() =

        CpuMaxPoolingLayer(this.name, this.numberRows, this.minimumColumns, this.maximumColumns)

}

fun maxPoolingLayer(numberRows : Int, numberColumns: Int) =

    maxPoolingLayer(null, numberRows, numberColumns)

fun maxPoolingLayer(name : String? = null, numberRows : Int, numberColumns: Int) =

    MaxPoolingLayer(name, numberRows, numberColumns, numberColumns)