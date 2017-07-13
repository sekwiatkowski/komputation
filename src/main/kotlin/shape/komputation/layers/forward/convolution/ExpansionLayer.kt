package shape.komputation.layers.forward.convolution

import shape.komputation.cpu.layers.forward.convolution.CpuExpansionLayer
import shape.komputation.layers.CpuForwardLayerInstruction


class ExpansionLayer(
    private val name : String?,
    private val filterWidth: Int,
    private val filterHeight: Int) : CpuForwardLayerInstruction {

    override fun buildForCpu() =

        CpuExpansionLayer(this.name, this.filterWidth, this.filterHeight)


}

fun expansionLayer(
    filterWidth: Int,
    filterHeight: Int) =

    expansionLayer(null, filterWidth, filterHeight)

fun expansionLayer(
    name : String?,
    filterWidth: Int,
    filterHeight: Int) =

    ExpansionLayer(name, filterWidth, filterHeight)