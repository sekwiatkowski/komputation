package shape.komputation.layers.forward.convolution

import shape.komputation.cpu.forward.convolution.CpuConvolutionalLayer
import shape.komputation.initialization.InitializationStrategy
import shape.komputation.layers.CpuForwardLayerInstruction
import shape.komputation.layers.concatenateNames
import shape.komputation.layers.forward.projection.projectionLayer
import shape.komputation.optimization.OptimizationStrategy

class ConvolutionalLayer(
    private val name : String?,
    private val numberFilters: Int,
    private val filterWidth: Int,
    private val filterHeight : Int,
    private val initializationStrategy : InitializationStrategy,
    private val optimizationStrategy : OptimizationStrategy? = null) : CpuForwardLayerInstruction {

    override fun buildForCpu(): CpuConvolutionalLayer {

        val expansionLayerName = concatenateNames(name, "expansion")
        val expansionLayer = expansionLayer(expansionLayerName, this.filterWidth, this.filterHeight).buildForCpu()

        val projectionLayerName = concatenateNames(name, "projection")
        val projectionLayer = projectionLayer(projectionLayerName, this.filterWidth * this.filterHeight, this.numberFilters, this.initializationStrategy, this.initializationStrategy, this.optimizationStrategy).buildForCpu()

        return CpuConvolutionalLayer(name, expansionLayer, projectionLayer)

    }


}

fun convolutionalLayer(
    numberFilters: Int,
    filterWidth: Int,
    filterHeight : Int,
    initializationStrategy : InitializationStrategy,
    optimizationStrategy : OptimizationStrategy? = null): ConvolutionalLayer {

    return convolutionalLayer(null, numberFilters, filterWidth, filterHeight, initializationStrategy, optimizationStrategy)

}

fun convolutionalLayer(
    name : String?,
    numberFilters: Int,
    filterWidth: Int,
    filterHeight : Int,
    initializationStrategy : InitializationStrategy,
    optimizationStrategy : OptimizationStrategy? = null) =

    ConvolutionalLayer(name, numberFilters, filterWidth, filterHeight, initializationStrategy, optimizationStrategy)