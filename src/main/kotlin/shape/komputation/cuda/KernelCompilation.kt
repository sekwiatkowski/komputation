package shape.komputation.cuda

import java.io.ByteArrayOutputStream
import java.io.IOException
import java.io.InputStream

fun compileKernel(cuFile: String, ptxFile: String, computeCapability : Pair<Int, Int>) {

    val (major, minor) = computeCapability

    val command = "nvcc -m64 -arch=compute_$major$minor -ptx $cuFile -o $ptxFile"

    val process = Runtime.getRuntime().exec(command)

    val exitValue : Int

    try {

        exitValue = process.waitFor()

    }
    catch (e: InterruptedException) {

        Thread.currentThread().interrupt()

        throw IOException("Interrupted while waiting for nvcc output", e)

    }

    if (exitValue != 0)
    {

        val errorMessage = String(toByteArray(process.errorStream))
        val outputMessage = String(toByteArray(process.inputStream))

        val report =
            """
                Could not create .ptx file:
                Exit value: $exitValue
                Output: $outputMessage
                Error: $errorMessage
            """.trimIndent()

        throw IOException(report)

    }

}

private fun toByteArray(inputStream: InputStream): ByteArray {

    val baos = ByteArrayOutputStream()
    val buffer = ByteArray(8192)

    while (true) {

        val read = inputStream.read(buffer)

        if (read == -1) {
            break
        }

        baos.write(buffer, 0, read)

    }

    return baos.toByteArray()

}
