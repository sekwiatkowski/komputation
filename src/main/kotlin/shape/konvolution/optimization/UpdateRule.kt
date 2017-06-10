package shape.konvolution.optimization

typealias UpdateRule = (indexRow : Int, indexColumn : Int, current : Double, derivative : Double) -> Double