package torch
package data

final case class Example[D1 <: DType, D2 <: DType](feature: Tensor[D1], target: Tensor[D2])
