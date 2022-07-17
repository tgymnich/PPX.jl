mutable struct NONE end

NONE(buf::AbstractVector{UInt8}) = FlatBuffers.read(NONE, buf)
NONE(io::IO) = FlatBuffers.deserialize(io, NONE)