package com.komputation.cuda.kernels.launch

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class RowwiseTest {

    private val maximumNumberThreadsPerBlock = 1024
    private val warpSize = 32

    @Test
    fun test1() {

        assertEquals(
            KernelLaunchConfiguration(1, this.warpSize, 1),
            computeRowwiseLaunchConfiguration(1, 1, this.warpSize, this.maximumNumberThreadsPerBlock)
        )

    }

    @Test
    fun test2() {

        assertEquals(
            KernelLaunchConfiguration(1, this.warpSize, 1),
            computeRowwiseLaunchConfiguration(1, 2, this.warpSize, this.maximumNumberThreadsPerBlock)
        )

    }

    @Test
    fun test3() {

        assertEquals(
            KernelLaunchConfiguration(1, this.maximumNumberThreadsPerBlock, 2),
            computeRowwiseLaunchConfiguration(1, 2048, this.warpSize, this.maximumNumberThreadsPerBlock)
        )

    }

    @Test
    fun test4() {

        assertEquals(
            KernelLaunchConfiguration(1, this.maximumNumberThreadsPerBlock, 2),
            computeRowwiseLaunchConfiguration(1, 1500, this.warpSize, this.maximumNumberThreadsPerBlock)
        )

    }


}