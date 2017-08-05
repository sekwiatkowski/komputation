package shape.komputation.layers.forward.convolution

import shape.komputation.cpu.functions.computeNumberFilterRowPositions
import shape.komputation.cpu.functions.computeNumberFilterColumnPositions
import shape.komputation.cpu.layers.forward.convolution.CpuConvolutionalLayer
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.forward.projection.projectionLayer
import shape.komputation.optimization.OptimizationInstruction

class ConvolutionalLayer(
    private val name : String?,
    private val numberInputRows : Int,
    private val numberInputColumns : Int,
    private val numberFilters: Int,
    private val filterWidth: Int,
    private val filterHeight : Int,
    private val weightInitialization: InitializationStrategy,
    private val biasInitialization: InitializationStrategy?,
    private val optimization: OptimizationInstruction? = null) : CpuForwardLayerInstruction {

    private val numberRowFilterPositions = computeNumberFilterRowPositions(this.numberInputRows, this.filterHeight)
    private val numberColumnFilterPositions = computeNumberFilterColumnPositions(this.numberInputColumns, this.filterWidth)
    private val numberConvolutions = this.numberRowFilterPositions * this.numberColumnFilterPositions

    override fun buildForCpu(): CpuConvolutionalLayer {

        val expansionLayerName = concatenateNames(name, "expansion")
        val expansionLayer = expansionLayer(
            expansionLayerName,
            this.numberInputRows,
            this.numberInputColumns,
            this.numberConvolutions,
            this.numberRowFilterPositions,
            this.filterWidth,
            this.filterHeight).buildForCpu()

        val projectionLayerName = concatenateNames(name, "projection")
        val projectionLayer = projectionLayer(projectionLayerName, this.filterWidth * this.filterHeight, this.numberConvolutions, this.numberFilters, this.weightInitialization, this.biasInitialization, this.optimization).buildForCpu()

        return CpuConvolutionalLayer(name, expansionLayer, projectionLayer)

    }


}

fun convolutionalLayer(
    numberInputRows : Int,
    numberInputColumns : Int,
    numberFilters: Int,
    filterWidth: Int,
    filterHeight : Int,
    weightInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy?,
    optimization: OptimizationInstruction? = null): ConvolutionalLayer {

    return convolutionalLayer(null, numberInputRows, numberInputColumns, numberFilters, filterWidth, filterHeight, weightInitialization, biasInitialization, optimization)

}

fun convolutionalLayer(
    name : String?,
    numberInputRows : Int,
    numberInputColumns : Int,
    numberFilters: Int,
    filterWidth: Int,
    filterHeight : Int,
    weightInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy?,
    optimization: OptimizationInstruction? = null) =

    ConvolutionalLayer(name, numberInputRows, numberInputColumns, numberFilters, filterWidth, filterHeight, weightInitialization, biasInitialization, optimization)