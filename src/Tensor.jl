using FlatBuffers

mutable struct Tensor
  data::Vector{Float64}
  shape::Vector{Int64}
end

Tensor(buf::AbstractVector{UInt8}) = FlatBuffers.read(Tensor, buf)
Tensor(io::IO) = FlatBuffers.deserialize(io, Tensor)