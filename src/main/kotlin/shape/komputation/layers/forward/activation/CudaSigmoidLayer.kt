package shape.komputation.layers.forward.activation

import jcuda.Pointer
import jcuda.driver.CUfunction
import jcuda.driver.JCudaDriver.cuCtxSynchronize
import jcuda.runtime.JCuda.cudaFree
import shape.komputation.cuda.*
import shape.komputation.layers.Resourceful
import shape.komputation.matrix.DoubleMatrix
import java.io.File

class CudaSigmoidLayer internal constructor(
    name : String? = null,
    private val inputDimension : Int,
    private val numberBlocks : Int,
    private val numberThreadsPerBlock : Int) : ActivationLayer(name), Resourceful {

    private val resultDimension = inputDimension

    private val function = CUfunction()
    private var ptxFile : File? = null

    private val deviceInput = Pointer()
    private val deviceResult = Pointer()

    private val parameters = Pointer.to(
        Pointer.to(intArrayOf(this.inputDimension)),
        Pointer.to(this.deviceInput),
        Pointer.to(this.deviceResult)
    )

    override fun acquire() {

        val cuFile = File(this.javaClass.getResource("/cuda/Sigmoid.cu").toURI())
        val cuPath = cuFile.path

        val ptxFile = File.createTempFile("sigmoid", ".ptx")
        ptxFile.deleteOnExit()
        val ptxPath = ptxFile.path
        this.ptxFile = ptxFile

        compileKernel(cuPath, ptxPath)

        loadKernel(ptxPath, this.function, "sigmoid_kernel")

        allocateDeviceMemory(this.deviceInput, this.inputDimension)
        allocateDeviceMemory(this.deviceResult, this.resultDimension)

    }

    override fun forward(input : DoubleMatrix, isTraining : Boolean): DoubleMatrix {

        setVector(this.deviceInput, input.entries, this.inputDimension)

        launchKernel(this.function, this.parameters, this.numberBlocks, this.numberThreadsPerBlock)

        cuCtxSynchronize()

        return DoubleMatrix(this.resultDimension, 1, getVector(this.deviceResult, this.resultDimension))

    }

    override fun backward(chain : DoubleMatrix) : DoubleMatrix {

        TODO()

    }

    override fun release() {

        this.ptxFile!!.delete()

        cudaFree(this.deviceInput)
        cudaFree(this.deviceResult)

    }

}
fun cudaSigmoidLayer(
    inputDimension : Int,
    numberBlocks : Int,
    numberThreadsPerBlock : Int) =

    cudaSigmoidLayer(null, inputDimension, numberBlocks, numberThreadsPerBlock)


fun cudaSigmoidLayer(
    name : String? = null,
    inputDimension : Int,
    numberBlocks : Int,
    numberThreadsPerBlock : Int) =

    CudaSigmoidLayer(name, inputDimension, numberBlocks, numberThreadsPerBlock)