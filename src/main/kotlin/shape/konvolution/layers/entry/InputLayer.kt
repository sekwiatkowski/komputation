package shape.konvolution.layers.entry

import shape.konvolution.matrix.RealMatrix

class InputLayer : EntryPoint() {

    override fun forward() {

        this.lastForwardResult = this.lastInput!! as RealMatrix

    }

}