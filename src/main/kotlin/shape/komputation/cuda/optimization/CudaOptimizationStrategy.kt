package shape.komputation.cuda.optimization

typealias CudaOptimizationStrategy = (numberParameters : Int, numberRows : Int, numberColumns : Int) -> CudaUpdateRule