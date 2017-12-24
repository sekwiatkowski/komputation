package com.komputation.instructions

interface HasOutputDimensions {
    val minimumNumberOutputColumns: Int
    val maximumNumberOutputColumns: Int
    val numberOutputRows : Int
}