package com.komputation.instructions.continuation.projection

import com.komputation.cpu.instructions.CpuContinuationInstruction
import com.komputation.cpu.layers.continuation.projection.CpuProjection
import com.komputation.cuda.CudaContext
import com.komputation.cuda.instructions.CudaContinuationInstruction
import com.komputation.cuda.layers.continuation.projection.CublasProjection
import com.komputation.initialization.InitializationStrategy
import com.komputation.instructions.concatenateNames
import com.komputation.instructions.continuation.BaseHigherOrderInstruction
import com.komputation.optimization.OptimizationInstruction
import jcuda.jcublas.cublasHandle

class Projection internal constructor(
    private val name : String?,
    private val outputDimension: Int,
    private val weightInitializationStrategy: InitializationStrategy,
    private val biasInitializationStrategy: InitializationStrategy?,
    private val optimizationStrategy : OptimizationInstruction? = null) : BaseHigherOrderInstruction(), CpuContinuationInstruction, CudaContinuationInstruction {

    private val weightingLayer = weighting(concatenateNames(name, "weighting"), this.outputDimension, this.weightInitializationStrategy, this.optimizationStrategy)

    private val biasLayer = if(this.biasInitializationStrategy != null) bias(concatenateNames(name, "bias"), this.biasInitializationStrategy, this.optimizationStrategy) else null

    override fun getLayers() = if(this.biasLayer != null) arrayOf(this.weightingLayer, this.biasLayer) else arrayOf(this.weightingLayer)

    override fun buildForCpu() =
        CpuProjection(this.name, this.weightingLayer.buildForCpu(), this.biasLayer?.buildForCpu())

    override fun buildForCuda(context: CudaContext, cublasHandle : cublasHandle) =
        CublasProjection(this.name, this.weightingLayer.buildForCuda(context, cublasHandle), this.biasLayer?.buildForCuda(context, cublasHandle))

}

fun projection(
    outputDimension: Int,
    initializationStrategy: InitializationStrategy,
    optimizationStrategy : OptimizationInstruction? = null) =
    projection(outputDimension, initializationStrategy, initializationStrategy, optimizationStrategy)

fun projection(
    outputDimension: Int,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    optimizationStrategy : OptimizationInstruction? = null) =
    projection(null, outputDimension, weightInitializationStrategy, biasInitializationStrategy, optimizationStrategy)

fun projection(
    name : String,
    outputDimension: Int,
    initializationStrategy: InitializationStrategy,
    optimizationStrategy : OptimizationInstruction? = null) =
    Projection(
        name,
        outputDimension,
        initializationStrategy,
        initializationStrategy,
        optimizationStrategy)

fun projection(
    name : String?,
    outputDimension: Int,
    weightInitializationStrategy: InitializationStrategy,
    biasInitializationStrategy: InitializationStrategy?,
    optimizationStrategy : OptimizationInstruction? = null) =
    Projection(
        name,
        outputDimension,
        weightInitializationStrategy,
        biasInitializationStrategy,
        optimizationStrategy)
