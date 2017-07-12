package shape.komputation.optimization

import jcuda.jcublas.cublasHandle

typealias CublasOptimizationStrategy = (cublasHandle : cublasHandle, numberRows : Int, numberColumns : Int) -> CublasUpdateRule