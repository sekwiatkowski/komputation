package shape.komputation.cuda.layers.forward.dropout

import jcuda.Pointer
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cpu.functions.seed
import shape.komputation.cuda.Kernel
import shape.komputation.cuda.allocateDeviceFloatMemory
import shape.komputation.cuda.layers.forward.activation.BaseCudaActivationLayer
import shape.komputation.cuda.setIntArray
import shape.komputation.layers.Resourceful
import shape.komputation.matrix.IntMath
import java.util.*

class CudaDropoutLayer internal constructor(
    name : String? = null,
    private val createTrainingKernel: () -> Kernel,
    private val createRuntimeKernel: () -> Kernel,
    private val createBackwardKernel: () -> Kernel,
    maximumThreadsPerBlock : Int,
    private val numberEntries: Int,
    private val random: Random,
    keepProbability : Float) : BaseCudaActivationLayer(name), Resourceful {

    private val numberThreads = Math.min(this.numberEntries, maximumThreadsPerBlock)
    private val numberBlocks = IntMath.ceil(this.numberEntries.toDouble() / this.numberThreads.toDouble())

    private val pointerToNumberEntries = Pointer.to(intArrayOf(this.numberEntries))
    private val pointerToKeepProbability = Pointer.to(floatArrayOf(keepProbability))
    private val pointerToDropoutProbability = Pointer.to(floatArrayOf(1.0f - keepProbability))

    private val deviceSeeds = Pointer()
    private val pointerToDeviceSeeds = Pointer.to(this.deviceSeeds)

    private val deviceMasks = Pointer()
    private val pointerToDeviceMasks = Pointer.to(this.deviceMasks)

    private var trainingKernel : Kernel? = null
    private var runtimeKernel : Kernel? = null
    private val deviceForwardResults = Pointer()
    private val pointerToDeviceForwardResults = Pointer.to(this.deviceForwardResults)

    private var backwardKernel : Kernel? = null
    private val deviceBackwardResults = Pointer()
    private val pointerToDeviceBackwardResults = Pointer.to(this.deviceBackwardResults)

    override fun acquire(maximumBatchSize : Int) {

        this.trainingKernel = this.createTrainingKernel()
        this.runtimeKernel = this.createRuntimeKernel()

        val seeds = IntArray(this.numberEntries)
        seed(this.random, seeds, this.numberEntries)

        setIntArray(seeds, this.numberEntries, this.deviceSeeds)

        allocateDeviceFloatMemory(this.deviceMasks, this.numberEntries)
        allocateDeviceFloatMemory(this.deviceForwardResults, this.numberEntries)

        this.backwardKernel = this.createBackwardKernel()
        allocateDeviceFloatMemory(this.deviceBackwardResults, this.numberEntries)

    }

    override fun forward(input : Pointer, batchSize : Int, isTraining : Boolean): Pointer {

        val pointerToInput = Pointer.to(input)

        if(isTraining) {

            this.trainingKernel!!.launch(
                Pointer.to(
                    this.pointerToNumberEntries,
                    this.pointerToDropoutProbability,
                    pointerToInput,
                    this.pointerToDeviceSeeds,
                    this.pointerToDeviceMasks,
                    this.pointerToDeviceForwardResults
                ),
                this.numberBlocks,
                batchSize,
                this.numberThreads,
                0
            )

        }
        else {

            this.runtimeKernel!!.launch(
                Pointer.to(
                    this.pointerToNumberEntries,
                    this.pointerToKeepProbability,
                    pointerToInput,
                    this.pointerToDeviceForwardResults
                ),
                this.numberBlocks,
                batchSize,
                this.numberThreads,
                0
            )

        }

        return this.deviceForwardResults

    }


    override fun backward(chain : Pointer, batchSize : Int) : Pointer {

        this.backwardKernel!!.launch(
            Pointer.to(
                this.pointerToNumberEntries,
                Pointer.to(chain),
                this.pointerToDeviceMasks,
                this.pointerToDeviceBackwardResults
            ),
            this.numberBlocks,
            batchSize,
            this.numberThreads,
            0
        )

        return this.deviceBackwardResults

    }

    override fun release() {

        this.trainingKernel!!.destroy()
        cudaFree(this.deviceBackwardResults)

        cudaFree(this.deviceForwardResults)
        cudaFree(this.deviceMasks)
        cudaFree(this.deviceSeeds)

        this.backwardKernel!!.destroy()
        this.runtimeKernel!!.destroy()

    }

}