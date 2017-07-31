package shape.komputation.cuda

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class KernelLaunchConfigurationTest {

    val numberMultiProcessors = 2
    val maximumNumberThreadsPerBlock = 1024
    val warpSize = 32
    val numberResidentWarps = 64

    @Test
    fun testEntrywise1() {

        assertEquals(
            EntrywiseLaunchConfiguration(1, 32, 1),
            computeEntrywiseLaunchConfiguration(1, numberMultiProcessors, numberResidentWarps, warpSize, maximumNumberThreadsPerBlock))

    }

    @Test
    fun testEntrywise2() {

        assertEquals(
            EntrywiseLaunchConfiguration(1, 32, 1),
            computeEntrywiseLaunchConfiguration(2, numberMultiProcessors, numberResidentWarps, warpSize, maximumNumberThreadsPerBlock))

    }

    @Test
    fun testEntrywise3() {

        assertEquals(
            EntrywiseLaunchConfiguration(2, 32, 1),
            computeEntrywiseLaunchConfiguration(33, numberMultiProcessors, numberResidentWarps, warpSize, maximumNumberThreadsPerBlock))

    }

    @Test
    fun testEntrywise4() {

        assertEquals(
            EntrywiseLaunchConfiguration(2, 512, 1),
            computeEntrywiseLaunchConfiguration(1024, numberMultiProcessors, numberResidentWarps, warpSize, maximumNumberThreadsPerBlock))

    }

    @Test
    fun testEntrywise5() {

        assertEquals(
            EntrywiseLaunchConfiguration(4, maximumNumberThreadsPerBlock, 3),
            computeEntrywiseLaunchConfiguration(10_000, numberMultiProcessors, numberResidentWarps, warpSize, maximumNumberThreadsPerBlock))

    }

    @Test
    fun testColumnwise1() {

        assertEquals(
            ColumnwiseLaunchConfiguration(1, 1, 1, computeDeviceFloatArraySize(1).toInt())  ,
            computeColumnwiseLaunchConfiguration(1, 1, maximumNumberThreadsPerBlock)
        )

    }

    @Test
    fun testColumnwise2() {

        assertEquals(
            ColumnwiseLaunchConfiguration(1, 2, 1, computeDeviceFloatArraySize(2).toInt())  ,
            computeColumnwiseLaunchConfiguration(1, 2, maximumNumberThreadsPerBlock)
        )

    }

    @Test
    fun testColumnwise3() {

        assertEquals(
            ColumnwiseLaunchConfiguration(1, 64, 1, computeDeviceFloatArraySize(64).toInt())  ,
            computeColumnwiseLaunchConfiguration(1, 33, maximumNumberThreadsPerBlock)
        )

    }

    @Test
    fun testColumnwise4() {

        assertEquals(
            ColumnwiseLaunchConfiguration(2, 1, 1, computeDeviceFloatArraySize(1).toInt())  ,
            computeColumnwiseLaunchConfiguration(2, 1, maximumNumberThreadsPerBlock)
        )

    }

    @Test
    fun testColumnwise5() {

        assertEquals(
            ColumnwiseLaunchConfiguration(2, 64, 1, computeDeviceFloatArraySize(64).toInt())  ,
            computeColumnwiseLaunchConfiguration(2, 33, maximumNumberThreadsPerBlock)
        )

    }

    @Test
    fun testColumnwise6() {

        assertEquals(
            ColumnwiseLaunchConfiguration(1, maximumNumberThreadsPerBlock, 2, computeDeviceFloatArraySize(maximumNumberThreadsPerBlock).toInt())  ,
            computeColumnwiseLaunchConfiguration(1, maximumNumberThreadsPerBlock+1, maximumNumberThreadsPerBlock)
        )

    }

    @Test
    fun testColumnwise7() {

        assertEquals(
            ColumnwiseLaunchConfiguration(2, maximumNumberThreadsPerBlock, 2, computeDeviceFloatArraySize(maximumNumberThreadsPerBlock).toInt())  ,
            computeColumnwiseLaunchConfiguration(2, maximumNumberThreadsPerBlock+1, maximumNumberThreadsPerBlock)
        )

    }

}