package com.komputation.instructions.continuation.convolution

import com.komputation.cpu.instructions.CpuContinuationInstruction
import com.komputation.cpu.layers.continuation.convolution.CpuConvolution
import com.komputation.cuda.CudaContext
import com.komputation.cuda.instructions.CudaContinuationInstruction
import com.komputation.cuda.layers.continuation.convolution.CudaConvolution
import com.komputation.initialization.InitializationStrategy
import com.komputation.instructions.concatenateNames
import com.komputation.instructions.continuation.BaseHigherOrderInstruction
import com.komputation.instructions.continuation.projection.projection
import com.komputation.optimization.OptimizationInstruction
import jcuda.jcublas.cublasHandle

class Convolution internal constructor(
    private val name : String?,
    private val numberFilters : Int,
    private val filterWidth: Int,
    private val filterHeight : Int,
    private val weightInitialization: InitializationStrategy,
    private val biasInitialization: InitializationStrategy?,
    private val optimization: OptimizationInstruction? = null) : BaseHigherOrderInstruction(), CpuContinuationInstruction, CudaContinuationInstruction {

    private val expansionLayer = expansion(
        concatenateNames(this.name, "expansion"),
        this.filterWidth,
        this.filterHeight)

    private val projectionLayer = projection(
        concatenateNames(this.name, "projection"),
        this.numberFilters,
        this.weightInitialization,
        this.biasInitialization,
        this.optimization)

    private val maxPoolingLayer = MaxPooling(concatenateNames(this.name, "max-pooling"))

    override fun getLayers() = arrayOf(this.expansionLayer, this.projectionLayer, this.maxPoolingLayer)

    override fun buildForCpu() =
        CpuConvolution(
            this.name,
            this.expansionLayer.buildForCpu(),
            this.projectionLayer.buildForCpu(),
            this.maxPoolingLayer.buildForCpu()
        )

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle) =
        CudaConvolution(
            this.name,
            this.expansionLayer.buildForCuda(context, cublasHandle),
            this.projectionLayer.buildForCuda(context, cublasHandle),
            this.maxPoolingLayer.buildForCuda(context, cublasHandle))

}

fun convolution(
    numberFilters: Int,
    filterWidth: Int,
    filterHeight : Int,
    initialization: InitializationStrategy,
    optimization: OptimizationInstruction? = null) =
    convolution(numberFilters, filterWidth, filterHeight, initialization, initialization, optimization)

fun convolution(
    numberFilters: Int,
    filterWidth: Int,
    filterHeight : Int,
    weightInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy? = weightInitialization,
    optimization: OptimizationInstruction? = null) =
    convolution(null, numberFilters, filterWidth, filterHeight, weightInitialization, biasInitialization, optimization)

fun convolution(
    name : String,
    numberFilters: Int,
    filterWidth: Int,
    filterHeight : Int,
    initialization: InitializationStrategy,
    optimization: OptimizationInstruction? = null) =
    convolution(name, numberFilters, filterWidth, filterHeight, initialization, initialization, optimization)

fun convolution(
    name : String?,
    numberFilters: Int,
    filterWidth: Int,
    filterHeight : Int,
    weightInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy? = weightInitialization,
    optimization: OptimizationInstruction? = null) =
    Convolution(name, numberFilters, filterWidth, filterHeight, weightInitialization, biasInitialization, optimization)