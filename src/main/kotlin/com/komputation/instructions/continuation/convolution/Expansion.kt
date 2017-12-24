package com.komputation.instructions.continuation.convolution

import com.komputation.cpu.functions.computeNumberFilterColumnPositions
import com.komputation.cpu.functions.computeNumberFilterRowPositions
import com.komputation.cpu.instructions.CpuContinuationInstruction
import com.komputation.cpu.layers.continuation.convolution.CpuExpansion
import com.komputation.cuda.CudaContext
import com.komputation.cuda.instructions.CudaContinuationInstruction
import com.komputation.cuda.kernels.ForwardKernels
import com.komputation.cuda.layers.continuation.convolution.CudaExpansion
import jcuda.jcublas.cublasHandle

fun computeNumberExpandedColumns(inputLength : Int, filterWidth: Int, numberFilterRowPositions: Int) =
    computeNumberFilterColumnPositions(inputLength, filterWidth) * numberFilterRowPositions

class Expansion internal constructor(
    private val name : String?,
    private val filterWidth: Int,
    private val filterHeight: Int) : CpuContinuationInstruction, CudaContinuationInstruction {

    private val filterLength = this.filterWidth * this.filterHeight

    private var numberInputRows = -1
    private val minimumNumberInputColumns = this.filterWidth
    private var maximumNumberInputColumns = -1

    override fun setInputDimensionsFromPreviousInstruction(numberInputRows: Int, minimumNumberInputColumns: Int, maximumNumberInputColumns: Int) {
        this.numberInputRows = numberInputRows
        this.maximumNumberInputColumns = maximumNumberInputColumns
    }

    override val numberOutputRows
        get() = this.filterLength
    override val minimumNumberOutputColumns
        get() = this.numberFilterRowPositions
    override val maximumNumberOutputColumns
        get() = computeNumberFilterColumnPositions(this.maximumNumberInputColumns, this.filterWidth) * this.numberFilterRowPositions

    private val numberFilterRowPositions
        get() = computeNumberFilterRowPositions(this.numberInputRows, this.filterHeight)

    override fun buildForCpu() =
        CpuExpansion(
            this.name,
            this.numberInputRows,
            this.minimumNumberInputColumns,
            this.maximumNumberInputColumns,
            this.numberFilterRowPositions,
            this.filterLength,
            this.filterWidth,
            this.filterHeight)

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle) =
        CudaExpansion(
            this.name,
            this.numberInputRows,
            this.maximumNumberInputColumns,
            this.numberFilterRowPositions,
            this.filterHeight,
            this.filterWidth,
            { context.createKernel(ForwardKernels.expansion()) },
            { context.createKernel(ForwardKernels.backwardExpansion()) },
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

}

fun expansion(
    filterWidth: Int,
    filterHeight: Int) =
    expansion(null, filterWidth, filterHeight)

fun expansion(
    name : String?,
    filterWidth: Int,
    filterHeight: Int) =
    Expansion(name, filterWidth, filterHeight)