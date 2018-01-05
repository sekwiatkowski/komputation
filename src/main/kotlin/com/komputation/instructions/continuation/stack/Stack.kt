package com.komputation.instructions.continuation.stack

import com.komputation.cpu.instructions.CpuContinuationInstruction
import com.komputation.cpu.layers.continuation.stack.CpuStack
import com.komputation.cuda.CudaContext
import com.komputation.cuda.instructions.CudaContinuationInstruction
import com.komputation.cuda.kernels.ArrayKernels
import com.komputation.cuda.layers.continuation.stack.CudaStack
import com.komputation.instructions.ContinuationInstruction
import jcuda.jcublas.cublasHandle

class Stack internal constructor(
    private val name : String?,
    private val continuationInstructions: Array<out ContinuationInstruction>) : CpuContinuationInstruction, CudaContinuationInstruction {

    override fun setInputDimensionsFromPreviousInstruction(numberInputRows: Int, minimumNumberInputColumns: Int, maximumNumberInputColumns: Int) {
        this.numberInputRows = numberInputRows
        this.maximumNumberInputColumns = maximumNumberInputColumns

        this.continuationInstructions.forEach { layer ->
            layer.setInputDimensionsFromPreviousInstruction(numberInputRows, minimumNumberInputColumns, maximumNumberInputColumns)
        }
    }

    private var numberInputRows = -1
    private var maximumNumberInputColumns = -1
    override val numberOutputRows
        get() = this.continuationInstructions.sumBy { it.numberOutputRows }
    override val minimumNumberOutputColumns
        get() = this.continuationInstructions[0].minimumNumberOutputColumns
    override val maximumNumberOutputColumns
        get() = this.continuationInstructions[0].maximumNumberOutputColumns

    override fun buildForCpu() =
        CpuStack(
            this.name,
            this.continuationInstructions.map { instruction -> (instruction as CpuContinuationInstruction).buildForCpu() }.toTypedArray())

    override fun buildForCuda(context: CudaContext, cublasHandle: cublasHandle) =
        CudaStack(
            this.name,
            this.numberInputRows,
            IntArray(this.continuationInstructions.size) { index -> this.continuationInstructions[index].numberOutputRows },
            this.maximumNumberInputColumns,
            this.maximumNumberOutputColumns,
            { context.createKernel(ArrayKernels.copyBlock()) },
            { context.createKernel(ArrayKernels.add()) },
            this.continuationInstructions.map { instruction -> (instruction as CudaContinuationInstruction).buildForCuda(context, cublasHandle) }.toTypedArray(),
            context.numberMultiprocessors,
            context.maximumNumberOfResidentWarpsPerMultiprocessor,
            context.warpSize,
            context.maximumNumberOfThreadsPerBlock)

}

fun stack(vararg continuations: ContinuationInstruction) =
    stack(null, *continuations)

fun stack(name : String?, vararg continuations: ContinuationInstruction) =
    Stack(name, continuations)