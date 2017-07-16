package shape.komputation.cuda.layers.forward.projection

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.CUfunction
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.acquireKernel
import shape.komputation.cuda.allocateDeviceMemory
import shape.komputation.cuda.functions.cublasBackwardProjectionWrtInput
import shape.komputation.cuda.functions.cublasBackwardProjectionWrtWeights
import shape.komputation.cuda.launchKernel
import shape.komputation.cuda.layers.BaseCudaForwardLayer
import shape.komputation.cuda.optimization.CudaUpdateRule
import shape.komputation.cuda.setVector
import shape.komputation.layers.Resourceful
import shape.komputation.optimization.Optimizable
import java.io.File

class CudaProjectionLayer internal constructor(
    name: String?,
    private val computeCapabilities: Pair<Int, Int>,
    private val maximumThreadsPerBlock: Int,
    private val cublasHandle: cublasHandle,
    private val initialWeights: DoubleArray,
    private val numberWeightRows: Int,
    private val numberWeightColumns: Int,
    private val weightUpdateRule: CudaUpdateRule? = null,

    private val initialBias: DoubleArray? = null,
    private val biasUpdateRule: CudaUpdateRule? = null) : BaseCudaForwardLayer(name), Optimizable, Resourceful {

    private val numberWeightEntries = this.numberWeightRows * this.numberWeightColumns

    private val hasBias = initialBias != null
    private val numberBiasEntries = if(this.hasBias) this.initialBias!!.size else 0

    private val inputDimension = this.numberWeightColumns
    private val resultDimension = this.numberWeightRows

    /*
                       i_1
                       i_2
                       i_3
        w_11 w_12 w_13
        w_21 w_22 w_23

        input dimension = number of weight columns
        result dimension = number of weight rows
    */

    // Weights
    private var deviceWeights = Pointer()
    private var pointerToDeviceWeights = Pointer.to(this.deviceWeights)
    private var deviceWeightGradientAccumulator = Pointer()
    private var pointerToDeviceWeightGradientAccumulator = Pointer.to(this.deviceWeightGradientAccumulator)

    // Bias
    private var deviceBias = Pointer()
    private var pointerToDeviceBias = Pointer.to(this.deviceBias)
    private var deviceBiasGradientAccumulator = Pointer()
    private var pointerToDeviceBiasGradientAccumulator = Pointer.to(this.deviceBiasGradientAccumulator)

    private var deviceInput = Pointer()

    // Result
    private var deviceResult = Pointer()
    private var pointerToDeviceResult = Pointer.to(this.deviceResult)

    // Gradient w.r.t input
    private var deviceBackwardWrtInput = Pointer()

    private val pointerToNumberWeightRows = Pointer.to(intArrayOf(this.numberWeightRows))
    private val pointerToNumberWeightColumns = Pointer.to(intArrayOf(this.numberWeightColumns))

    // Forward kernels
    private val projectionKernel = CUfunction()
    private var projectionPtxFile : File? = null

    private val projectionWithBiasKernel = CUfunction()
    private var projectionWithBiasPtxFile : File? = null

    private val accumulationKernel = CUfunction()
    private var accumulationPtxFile : File? = null

    private val blockSize = 32
    private val numberBlocks = (this.numberWeightRows + this.blockSize - 1) / this.blockSize
    private val sharedMemoryBytes = this.blockSize * Sizeof.DOUBLE

    override fun acquire() {

        setVector(this.initialWeights, this.numberWeightEntries, this.deviceWeights)

        if(this.initialBias != null) {

            setVector(this.initialBias, this.numberBiasEntries, this.deviceBias)

        }

        allocateDeviceMemory(this.deviceResult, this.numberWeightRows)

        allocateDeviceMemory(this.deviceBackwardWrtInput, this.inputDimension)
        allocateDeviceMemory(this.deviceWeightGradientAccumulator, this.numberWeightEntries)
        allocateDeviceMemory(this.deviceBiasGradientAccumulator, this.numberBiasEntries)

        if (this.weightUpdateRule is Resourceful) {

            this.weightUpdateRule.acquire()

        }

        if (this.biasUpdateRule is Resourceful) {

            this.biasUpdateRule.acquire()

        }

        this.projectionPtxFile = acquireKernel(
            File(this.javaClass.getResource("/cuda/projection/Projection.cu").toURI()),
            "projectionKernel",
            this.projectionKernel,
            this.computeCapabilities)

        this.projectionWithBiasPtxFile = acquireKernel(
            File(this.javaClass.getResource("/cuda/projection/ProjectionWithBias.cu").toURI()),
            "projectionWithBiasKernel",
            this.projectionWithBiasKernel,
            this.computeCapabilities)

        this.accumulationPtxFile = acquireKernel(
            File(this.javaClass.getResource("/cuda/accumulation/Accumulation.cu").toURI()),
            "accumulationKernel",
            this.accumulationKernel,
            this.computeCapabilities)


    }

    override fun forward(input : Pointer): Pointer {

        this.deviceInput = input

        val pointerToDeviceInput = Pointer.to(deviceInput)

        if (this.hasBias) {

            val parameters = Pointer.to(
                pointerToDeviceInput,
                this.pointerToDeviceWeights,
                this.pointerToNumberWeightRows,
                this.pointerToNumberWeightColumns,
                this.pointerToDeviceBias,
                this.pointerToDeviceResult
            )

            launchKernel(
                this.projectionWithBiasKernel,
                parameters,
                this.numberBlocks,
                this.blockSize,
                this.sharedMemoryBytes
            )

        }
        else {

            val parameters = Pointer.to(
                pointerToDeviceInput,
                this.pointerToDeviceWeights,
                this.pointerToNumberWeightRows,
                this.pointerToNumberWeightColumns,
                this.pointerToDeviceResult
            )

            launchKernel(
                this.projectionKernel,
                parameters,
                this.numberBlocks,
                this.blockSize,
                this.sharedMemoryBytes
            )

        }

        return this.deviceResult

    }

    /*
                          x_1
                          x_2
                          x_3
        w_11 w_12 w_13    w_11 * x_1 + w_12 * x_2 + w_13 * x_3
        w_21 w_22 w_23    w_21 * x_1 + w_22 * x_2 + w_23 * x_3

        Differentiation w.r.t input:

        d Wx / d x = w_11 + w_21
                     w_12 + w_22
                     w_13 + w_23

        gemv solution:
                                  chain_1
                                  chain_2
        transposed W >> w_11 w_21
                        w_12 w_22
                        w_13 w_23

        Differentiation w.r.t weights:

        d Wx / d W = x_1 x_2 x_3
                     x_1 x_2 x_3

        ger solution:
                x1 x2 x3 << transposed x
        chain_1
        chain_2

     */

    private val numberBiasBlocks = (this.numberBiasEntries + this.maximumThreadsPerBlock - 1) / this.maximumThreadsPerBlock
    private val numberBiasThreadsPerBlock = if(this.numberBiasBlocks == 1) this.numberBiasEntries else this.maximumThreadsPerBlock
    private val pointerToNumberBiasEntries = Pointer.to(intArrayOf(this.numberBiasEntries))

    override fun backward(chain: Pointer): Pointer {

        cublasBackwardProjectionWrtInput(this.cublasHandle, this.deviceWeights, this.numberWeightRows, this.numberWeightColumns, chain, this.deviceBackwardWrtInput)

        cublasBackwardProjectionWrtWeights(this.cublasHandle, this.deviceInput, chain, this.deviceWeightGradientAccumulator, this.numberWeightRows, this.numberWeightColumns)

        if (this.initialBias != null) {

            launchKernel(
                this.accumulationKernel,
                Pointer.to(
                    this.pointerToNumberBiasEntries,
                    this.pointerToDeviceBiasGradientAccumulator,
                    Pointer.to(chain)
                ),
                this.numberBiasBlocks,
                this.numberBiasThreadsPerBlock,
                0
            )

            // cublasBackwardProjectionWrtBias(this.cublasHandle, chain, this.chainDimension, this.deviceBiasGradientAccumulator)

        }

        return this.deviceBackwardWrtInput

    }

    override fun optimize(scalingFactor: Double) {

        this.weightUpdateRule?.update(this.pointerToDeviceWeights, scalingFactor, this.pointerToDeviceWeightGradientAccumulator)

        this.biasUpdateRule?.update(this.pointerToDeviceBias, scalingFactor, this.pointerToDeviceBiasGradientAccumulator)

    }

    override fun release() {

        if (this.weightUpdateRule is Resourceful) {

            this.weightUpdateRule.release()

        }
        cudaFree(this.deviceWeightGradientAccumulator)
        cudaFree(this.deviceWeights)

        if(this.initialBias != null) {

            if (this.biasUpdateRule is Resourceful) {

                this.biasUpdateRule.release()

            }
            cudaFree(this.deviceBiasGradientAccumulator)
            cudaFree(this.deviceBias)

        }

        cudaFree(this.deviceResult)

        cudaFree(this.deviceBackwardWrtInput)

        this.projectionPtxFile!!.deleteOnExit()
        this.projectionWithBiasPtxFile!!.deleteOnExit()
        this.accumulationPtxFile!!.deleteOnExit()

    }

}