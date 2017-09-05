package shape.komputation.cuda.kernels

object HashtableKernels {

    fun groupSum() = KernelInstruction(
        "groupSumKernel",
        "groupSumKernel",
        "hashtable/GroupSumKernel.cu")

    fun hash() = KernelInstruction(
        "hashKernel",
        "hashKernel",
        "hashtable/HashKernel.cu")

}