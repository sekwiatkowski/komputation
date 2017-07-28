package shape.komputation.cuda.optimization

import jcuda.Pointer
import shape.komputation.cuda.Kernel
import shape.komputation.layers.Resourceful
import shape.komputation.matrix.IntMath

class CudaStochasticGradientDescent(
    private val createKernel: () -> Kernel,
    maximumThreadsPerBlock: Int,
    private val size : Int,
    private val learningRate: Float) : CudaUpdateRule, Resourceful {

    private val pointerToSize = Pointer.to(intArrayOf(this.size))
    private val pointerToLearningRate = Pointer.to(floatArrayOf(this.learningRate))

    private val numberThreads = Math.min(this.size, maximumThreadsPerBlock)
    private val numberBlocks = IntMath.ceil(this.size.toDouble() / numberThreads.toDouble())

    private var kernel : Kernel? = null

    override fun acquire(maximumBatchSize : Int) {

        this.kernel = this.createKernel()

    }

    private val scalingFactorArray = floatArrayOf(Float.NaN)
    private val pointerToScalingFactor = Pointer.to(this.scalingFactorArray)

    override fun update(pointerToDeviceParameter: Pointer, scalingFactor : Float, pointerToDeviceGradient: Pointer) {

        this.scalingFactorArray[0] = scalingFactor

        val parameters = Pointer.to(
            this.pointerToSize,
            this.pointerToScalingFactor,
            this.pointerToLearningRate,
            pointerToDeviceParameter,
            pointerToDeviceGradient
        )

        this.kernel!!.launch(
            parameters,
            this.numberBlocks,
            1,
            this.numberThreads,
            0
        )

    }

    override fun release() {

        this.kernel!!.destroy()

    }

}