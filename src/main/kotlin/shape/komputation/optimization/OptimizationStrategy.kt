package shape.komputation.optimization

import jcuda.jcublas.cublasHandle

typealias OptimizationStrategy = (numberRows : Int, numberColumns : Int) -> UpdateRule

typealias CublasOptimizationStrategy = (cublasHandle : cublasHandle, numberRows : Int, numberColumns : Int) -> CublasUpdateRule