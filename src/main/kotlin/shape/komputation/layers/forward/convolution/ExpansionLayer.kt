package shape.komputation.layers.forward.convolution

import shape.komputation.cpu.layers.forward.convolution.CpuExpansionLayer
import shape.komputation.layers.CpuForwardLayerInstruction


class ExpansionLayer(
    private val name : String?,
    private val numberInputRows : Int,
    private val numberInputColumns : Int,
    private val numberConvolutions : Int,
    private val numberFilterRowPositions: Int,
    private val filterWidth: Int,
    private val filterHeight: Int) : CpuForwardLayerInstruction {

    override fun buildForCpu() =

        CpuExpansionLayer(this.name, this.numberInputRows, this.numberInputColumns, this.numberConvolutions, this.numberFilterRowPositions, this.filterWidth, this.filterHeight)


}

fun expansionLayer(
    numberInputRows : Int,
    numberInputColumns : Int,
    numberConvolutions: Int,
    numberFilterRowPositions: Int,
    filterWidth: Int,
    filterHeight: Int) =

    expansionLayer(null, numberInputRows, numberInputColumns, numberConvolutions, numberFilterRowPositions, filterWidth, filterHeight)

fun expansionLayer(
    name : String?,
    numberInputRows : Int,
    numberInputColumns : Int,
    numberConvolutions: Int,
    numberFilterRowPositions: Int,
    filterWidth: Int,
    filterHeight: Int) =

    ExpansionLayer(name, numberInputRows, numberInputColumns, numberConvolutions, numberFilterRowPositions, filterWidth, filterHeight)