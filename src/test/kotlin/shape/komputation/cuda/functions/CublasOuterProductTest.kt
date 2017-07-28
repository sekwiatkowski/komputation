package shape.komputation.cuda.functions

import jcuda.Pointer
import jcuda.jcublas.JCublas2.cublasCreate
import jcuda.jcublas.JCublas2.cublasDestroy
import jcuda.jcublas.cublasHandle
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Test
import shape.komputation.cuda.allocateDeviceFloatMemory
import shape.komputation.cuda.getFloatArray
import shape.komputation.cuda.setFloatArray

class CublasOuterProductTest {

    @Test
    fun test() {

        val cublasHandle = cublasHandle()
        cublasCreate(cublasHandle)

        val firstPointer = Pointer()
        val firstVector = floatArrayOf(1.0f, 2.0f)
        val firstDimension = firstVector.size
        setFloatArray(firstVector, firstDimension, firstPointer)

        val secondPointer = Pointer()
        val secondVector = floatArrayOf(1.0f, 2.0f, 3.0f)
        val secondDimension = secondVector.size
        setFloatArray(secondVector, secondDimension, secondPointer)

        val resultPointer = Pointer()
        allocateDeviceFloatMemory(resultPointer, firstDimension * secondDimension)

        cublasOuterProduct(
            cublasHandle,
            firstDimension,
            firstPointer,
            secondDimension,
            secondPointer,
            resultPointer
        )

        val actual = getFloatArray(resultPointer, firstDimension * secondDimension)
        val expected = floatArrayOf(1.0f, 2.0f, 3.0f, 2.0f, 4.0f, 6.0f)

        cublasDestroy(cublasHandle)

        assertArrayEquals(expected, actual)

    }

}