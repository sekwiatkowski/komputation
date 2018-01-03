package com.komputation.instructions.continuation.dense

import com.komputation.cpu.instructions.CpuContinuationInstruction
import com.komputation.cpu.layers.continuation.dense.CpuDense
import com.komputation.cuda.CudaContext
import com.komputation.cuda.instructions.CudaContinuationInstruction
import com.komputation.cuda.layers.CudaContinuation
import com.komputation.cuda.layers.continuation.dense.CudaDense
import com.komputation.initialization.InitializationStrategy
import com.komputation.instructions.concatenateNames
import com.komputation.instructions.continuation.BaseHigherOrderInstruction
import com.komputation.instructions.continuation.activation.Activation
import com.komputation.instructions.continuation.activation.activation
import com.komputation.instructions.continuation.projection.projection
import com.komputation.optimization.OptimizationInstruction
import jcuda.jcublas.cublasHandle

class Dense internal constructor(
    private val name : String?,
    outputDimension: Int,
    weightInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy?,
    activation: Activation,
    optimization: OptimizationInstruction?) : BaseHigherOrderInstruction(), CpuContinuationInstruction, CudaContinuationInstruction {

    private val projection = projection(
        concatenateNames(name, "projection"),
        outputDimension,
        weightInitialization,
        biasInitialization,
        optimization)

    private val activation = activation(
        concatenateNames(name, "activation"),
        activation)

    override fun getLayers() = arrayOf(this.projection, this.activation)

    override fun buildForCpu(): CpuDense {
        val projectionLayer = this.projection.buildForCpu()

        val activationLayer = this.activation.buildForCpu()

        return CpuDense(this.name, projectionLayer, activationLayer)
    }

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle): CudaContinuation {
        val projectionLayer = this.projection.buildForCuda(context, cublasHandle)

        val activationLayer = this.activation.buildForCuda(context, cublasHandle)

        return CudaDense(this.name, projectionLayer, activationLayer)
    }

}

fun dense(
    outputDimension: Int,
    activation: Activation,
    initialization: InitializationStrategy,
    optimization: OptimizationInstruction? = null) =
    dense(
        outputDimension,
        activation,
        initialization,
        initialization,
        optimization)

fun dense(
    outputDimension: Int,
    activation: Activation,
    weightInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy? = weightInitialization,
    optimization: OptimizationInstruction? = null) =
    dense(
        null,
        outputDimension,
        activation,
        weightInitialization,
        biasInitialization,
        optimization)

fun dense(
    name: String,
    numberOutputRows: Int,
    activation: Activation,
    initialization: InitializationStrategy,
    optimization: OptimizationInstruction? = null) =
    dense(
        name,
        numberOutputRows,
        activation,
        initialization,
        initialization,
        optimization)

fun dense(
    name: String?,
    numberOutputRows: Int,
    activation: Activation,
    weightInitialization: InitializationStrategy,
    biasInitialization: InitializationStrategy? = weightInitialization,
    optimization: OptimizationInstruction? = null) =
    Dense(
        name,
        numberOutputRows,
        weightInitialization,
        biasInitialization,
        activation,
        optimization)