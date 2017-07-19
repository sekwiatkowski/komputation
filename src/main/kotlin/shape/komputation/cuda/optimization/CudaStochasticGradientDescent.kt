package shape.komputation.cuda.optimization

import jcuda.Pointer
import shape.komputation.cuda.Kernel
import shape.komputation.layers.Resourceful

class CudaStochasticGradientDescent(
    private val kernel: Kernel,
    maximumThreadsPerBlock: Int,
    private val size : Int,
    private val learningRate: Double) : CudaUpdateRule, Resourceful {

    private val pointerToSize = Pointer.to(intArrayOf(this.size))
    private val pointerToLearningRate = Pointer.to(doubleArrayOf(this.learningRate))

    private val numberThreads = Math.min(this.size, maximumThreadsPerBlock)
    private val numberBlocks = Math.ceil(this.size.toDouble() / numberThreads.toDouble()).toInt()

    override fun acquire() {

        this.kernel.acquire()

    }

    private val scalingFactorArray = doubleArrayOf(Double.NaN)
    private val pointerToScalingFactor = Pointer.to(this.scalingFactorArray)

    override fun update(pointerToDeviceParameter: Pointer, scalingFactor : Double, pointerToDeviceGradient: Pointer) {

        this.scalingFactorArray[0] = scalingFactor

        this.kernel.launch(
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

        this.kernel.release()

    }

}