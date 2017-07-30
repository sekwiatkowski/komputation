package shape.komputation.cuda

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class KernelLaunchConfigurationTest {

    val numberMultiProcessors = 2
    val maximumNumberThreadsPerBlock = 1024
    val warpSize = 32
    val numberResidentWarps = 64

    @Test
    fun test1() {

        assertEquals(
            Triple(1, 32, 1),
            computeKernelLaunchConfigurationForElementWiseFunctions(1, numberMultiProcessors, numberResidentWarps, warpSize, maximumNumberThreadsPerBlock))

    }

    @Test
    fun test2() {

        assertEquals(
            Triple(1, 32, 1),
            computeKernelLaunchConfigurationForElementWiseFunctions(2, numberMultiProcessors, numberResidentWarps, warpSize, maximumNumberThreadsPerBlock))

    }

    @Test
    fun test3() {

        assertEquals(
            Triple(2, 32, 1),
            computeKernelLaunchConfigurationForElementWiseFunctions(33, numberMultiProcessors, numberResidentWarps, warpSize, maximumNumberThreadsPerBlock))

    }

    @Test
    fun test4() {

        assertEquals(
            Triple(2, 512, 1),
            computeKernelLaunchConfigurationForElementWiseFunctions(1024, numberMultiProcessors, numberResidentWarps, warpSize, maximumNumberThreadsPerBlock))

    }

    @Test
    fun test5() {

        assertEquals(
            Triple(4, maximumNumberThreadsPerBlock, 3),
            computeKernelLaunchConfigurationForElementWiseFunctions(10_000, numberMultiProcessors, numberResidentWarps, warpSize, maximumNumberThreadsPerBlock))

    }

}