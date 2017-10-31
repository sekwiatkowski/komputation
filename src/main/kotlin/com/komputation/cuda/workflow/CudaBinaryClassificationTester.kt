package com.komputation.cuda.workflow

import com.komputation.cuda.allocateDeviceIntMemory
import com.komputation.cuda.getIntArray
import com.komputation.cuda.kernels.Kernel
import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree

class CudaBinaryClassificationTester(
    private val numberInstances : Int,
    private val numberColumns : Int,
    private val createKernel: () -> Kernel) : CudaClassificationTester {

    private val deviceCorrectPredictions = Pointer()
    private var kernel : Kernel? = null

    private val pointerToNumberColumns = Pointer.to(intArrayOf(this.numberColumns))
    private val pointerToCorrectPredictions = Pointer.to(this.deviceCorrectPredictions)

    private var count = 0

    override fun acquire(maximumBatchSize: Int) {

        allocateDeviceIntMemory(this.deviceCorrectPredictions, this.numberInstances)

        this.kernel = this.createKernel()

    }

    override fun evaluateBatch(batchSize: Int, pointerToPredictions : Pointer, pointerToTargets : Pointer) {

        val parameters = Pointer.to(
            Pointer.to(intArrayOf(this.count)),
            this.pointerToNumberColumns,
            pointerToPredictions,
            pointerToTargets,
            this.pointerToCorrectPredictions
        )

        this.kernel!!.launch(
            parameters,
            batchSize,
            1,
            1,
            0)

        this.count += batchSize

    }

    override fun resetCount() {

        this.count = 0

    }

    override fun computeAccuracy() =

        getIntArray(this.deviceCorrectPredictions, this.numberInstances).sum().toFloat().div(this.numberInstances.toFloat())

    override fun release() {

        this.kernel!!.destroy()

        cudaFree(this.deviceCorrectPredictions)

    }

}