package shape.komputation.cuda

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import shape.komputation.cuda.kernels.launch.KernelLaunchConfiguration
import shape.komputation.cuda.kernels.launch.computeColumnwiseLaunchConfiguration
import shape.komputation.cuda.kernels.launch.computeEntrywiseLaunchConfiguration

class KernelLaunchConfigurationTest {

    val numberMultiProcessors = 2
    val maximumNumberThreadsPerBlock = 1024
    val warpSize = 32
    val numberResidentWarps = 64

    @Test
    fun testEntrywise1() {

        assertEquals(
            KernelLaunchConfiguration(1, 32, 1),
            computeEntrywiseLaunchConfiguration(1, numberMultiProcessors, numberResidentWarps, warpSize, maximumNumberThreadsPerBlock))

    }

    @Test
    fun testEntrywise2() {

        assertEquals(
            KernelLaunchConfiguration(1, 32, 1),
            computeEntrywiseLaunchConfiguration(2, numberMultiProcessors, numberResidentWarps, warpSize, maximumNumberThreadsPerBlock))

    }

    @Test
    fun testEntrywise3() {

        assertEquals(
            KernelLaunchConfiguration(2, 32, 1),
            computeEntrywiseLaunchConfiguration(33, numberMultiProcessors, numberResidentWarps, warpSize, maximumNumberThreadsPerBlock))

    }

    @Test
    fun testEntrywise4() {

        assertEquals(
            KernelLaunchConfiguration(2, 512, 1),
            computeEntrywiseLaunchConfiguration(1024, numberMultiProcessors, numberResidentWarps, warpSize, maximumNumberThreadsPerBlock))

    }

    @Test
    fun testEntrywise5() {

        assertEquals(
            KernelLaunchConfiguration(4, maximumNumberThreadsPerBlock, 3),
            computeEntrywiseLaunchConfiguration(10_000, numberMultiProcessors, numberResidentWarps, warpSize, maximumNumberThreadsPerBlock))

    }

    @Test
    fun testColumnwise1() {

        assertEquals(
            KernelLaunchConfiguration(1, 1, 1),
            computeColumnwiseLaunchConfiguration(1, 1, maximumNumberThreadsPerBlock)
        )

    }

    @Test
    fun testColumnwise2() {

        assertEquals(
            KernelLaunchConfiguration(1, 2, 1),
            computeColumnwiseLaunchConfiguration(2, 1, maximumNumberThreadsPerBlock)
        )

    }

    @Test
    fun testColumnwise3() {

        assertEquals(
            KernelLaunchConfiguration(1, 33, 1),
            computeColumnwiseLaunchConfiguration(33, 1, maximumNumberThreadsPerBlock)
        )

    }

    @Test
    fun testColumnwise4() {

        assertEquals(
            KernelLaunchConfiguration(2, 1, 1),
            computeColumnwiseLaunchConfiguration(1, 2, maximumNumberThreadsPerBlock)
        )

    }

    @Test
    fun testColumnwise5() {

        assertEquals(
            KernelLaunchConfiguration(2, 33, 1),
            computeColumnwiseLaunchConfiguration(33, 2, maximumNumberThreadsPerBlock)
        )

    }

    @Test
    fun testColumnwise6() {

        assertEquals(
            KernelLaunchConfiguration(1, maximumNumberThreadsPerBlock, 2),
            computeColumnwiseLaunchConfiguration(maximumNumberThreadsPerBlock + 1, 1, maximumNumberThreadsPerBlock)
        )

    }

    @Test
    fun testColumnwise7() {

        assertEquals(
            KernelLaunchConfiguration(2, maximumNumberThreadsPerBlock, 2),
            computeColumnwiseLaunchConfiguration(maximumNumberThreadsPerBlock + 1, 2, maximumNumberThreadsPerBlock)
        )

    }

}