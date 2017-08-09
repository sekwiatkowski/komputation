package shape.komputation.layers.forward.convolution

import shape.komputation.cpu.layers.forward.convolution.CpuExpansionLayer
import shape.komputation.layers.CpuForwardLayerInstruction


class ExpansionLayer(
    private val name : String?,
    private val numberInputRows : Int,
    private val numberInputColumns : Int,
    private val hasFixedLength: Boolean,
    private val numberFilterRowPositions: Int,
    private val filterWidth: Int,
    private val filterHeight: Int) : CpuForwardLayerInstruction {

    private val minimumInputLength = if(this.hasFixedLength) this.numberInputColumns else this.filterWidth
    private val maximumInputLength = this.numberInputColumns

    private val filterLength = this.filterWidth * this.filterHeight

    override fun buildForCpu() =

        CpuExpansionLayer(
            this.name,
            this.numberInputRows,
            this.minimumInputLength,
            this.maximumInputLength,
            this.numberFilterRowPositions,
            this.filterLength,
            this.filterWidth,
            this.filterHeight)


}

fun expansionLayer(
    numberInputRows : Int,
    numberInputColumns : Int,
    hasFixedLength: Boolean,
    numberFilterRowPositions: Int,
    filterWidth: Int,
    filterHeight: Int) =

    expansionLayer(null, numberInputRows, numberInputColumns, hasFixedLength, numberFilterRowPositions, filterWidth, filterHeight)

fun expansionLayer(
    name : String?,
    numberInputRows : Int,
    numberInputColumns : Int,
    hasFixedLength: Boolean,
    numberFilterRowPositions: Int,
    filterWidth: Int,
    filterHeight: Int) =

    ExpansionLayer(name, numberInputRows, numberInputColumns, hasFixedLength, numberFilterRowPositions, filterWidth, filterHeight)