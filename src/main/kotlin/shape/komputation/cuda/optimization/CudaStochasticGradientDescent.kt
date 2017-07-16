package shape.komputation.cuda.optimization

import jcuda.Pointer
import jcuda.driver.CUfunction
import shape.komputation.cuda.acquireKernel
import shape.komputation.cuda.launchKernel
import shape.komputation.layers.Resourceful
import java.io.File

class CudaStochasticGradientDescent(
    private val capabilities : Pair<Int, Int>,
    maximumThreadsPerBlock: Int,
    private val size : Int,
    private val learningRate: Double) : CudaUpdateRule, Resourceful {

    private var ptxFile: File? = null
    private val kernel = CUfunction()

    private val pointerToSize = Pointer.to(intArrayOf(this.size))
    private val pointerToLearningRate = Pointer.to(doubleArrayOf(this.learningRate))

    private val numberThreads = Math.min(this.size, maximumThreadsPerBlock)
    private val numberBlocks = Math.ceil(this.size.toDouble() / numberThreads.toDouble()).toInt()

    override fun acquire() {

        this.ptxFile = acquireKernel(
            File(javaClass.getResource("/cuda/stochasticgradientdescent/StochasticGradientDescentKernel.cu").toURI()),
            "stochasticGradientDescentKernel",
            kernel,
            capabilities
        )

    }

    private val scalingFactorArray = doubleArrayOf(Double.NaN)
    private val pointerToScalingFactor = Pointer.to(this.scalingFactorArray)

    override fun update(pointerToDeviceParameter: Pointer, scalingFactor : Double, pointerToDeviceGradient: Pointer) {

        this.scalingFactorArray[0] = scalingFactor

        launchKernel(
            this.kernel,
            Pointer.to(
                this.pointerToSize,
                pointerToDeviceParameter,
                this.pointerToScalingFactor,
                this.pointerToLearningRate,
                pointerToDeviceGradient
            ),
            this.numberBlocks,
            this.numberThreads,
            0
        )

    }

    override fun release() {

        this.ptxFile!!.delete()

    }


}