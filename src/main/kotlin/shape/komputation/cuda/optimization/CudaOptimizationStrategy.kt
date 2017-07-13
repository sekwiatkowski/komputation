package shape.komputation.cuda.optimization

import jcuda.jcublas.cublasHandle

typealias CudaOptimizationStrategy = (cublasHandle : cublasHandle, numberRows : Int, numberColumns : Int) -> CublasUpdateRule