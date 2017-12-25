package com.komputation.cuda.layers

import com.komputation.cuda.CudaBackwardPropagation
import com.komputation.cuda.CudaForwardPropagation
import com.komputation.cuda.CudaLayer

interface CudaContinuation : CudaLayer, CudaForwardPropagation, CudaBackwardPropagation