using FlatBuffers

include("Tensor.jl")
include("Distribution.jl")
include("Utils.jl")

mutable struct Handshake
  system_name::String
end

Handshake(buf::AbstractVector{UInt8}) = FlatBuffers.read(Handshake, buf)
Handshake(io::IO) = FlatBuffers.deserialize(io, Handshake)

mutable struct HandshakeResult
  system_name::String
  model_name::String
end

HandshakeResult(buf::AbstractVector{UInt8}) = FlatBuffers.read(HandshakeResult, buf)
HandshakeResult(io::IO) = FlatBuffers.deserialize(io, HandshakeResult)

mutable struct Run end

Run(buf::AbstractVector{UInt8}) = FlatBuffers.read(Run, buf)
Run(io::IO) = FlatBuffers.deserialize(io, Run)

mutable struct RunResult
    result::Tensor
end

RunResult(buf::AbstractVector{UInt8}) = FlatBuffers.read(RunResult, buf)
RunResult(io::IO) = FlatBuffers.deserialize(io, RunResult)

@with_kw mutable struct Sample
    address::String
    name::String
    distribution_type::Int8
    distribution::Distribution
    control::Bool = true
end

Sample(buf::AbstractVector{UInt8}) = FlatBuffers.read(Sample, buf)
Sample(io::IO) = FlatBuffers.deserialize(io, Sample)

mutable struct SampleResult
    result::Tensor
end

SampleResult(buf::AbstractVector{UInt8}) = FlatBuffers.read(SampleResult, buf)
SampleResult(io::IO) = FlatBuffers.deserialize(io, SampleResult)

mutable struct Observe
    address::String
    name::String
    distribution_type::Int8
    distribution::Distribution
    value::Tensor
end

Observe(buf::AbstractVector{UInt8}) = FlatBuffers.read(Observe, buf)
Observe(io::IO) = FlatBuffers.deserialize(io, Observe)

mutable struct ObserveResult end

ObserveResult(buf::AbstractVector{UInt8}) = FlatBuffers.read(ObserveResult, buf)
ObserveResult(io::IO) = FlatBuffers.deserialize(io, ObserveResult)

mutable struct Tag
    address::String
    name::String
    value::Tensor
end

Tag(buf::AbstractVector{UInt8}) = FlatBuffers.read(Tag, buf)
Tag(io::IO) = FlatBuffers.deserialize(io, Tag)

mutable struct TagResult end

TagResult(buf::AbstractVector{UInt8}) = FlatBuffers.read(TagResult, buf)
TagResult(io::IO) = FlatBuffers.deserialize(io, TagResult)

mutable struct Reset end

Reset(buf::AbstractVector{UInt8}) = FlatBuffers.read(Reset, buf)
Reset(io::IO) = FlatBuffers.deserialize(io, Reset)

FlatBuffers.@UNION(MessageBody, (
  NONE,
  Handshake,
  HandshakeResult,
  Run,
  RunResult,
  Sample,
  SampleResult,
  Observe,
  ObserveResult,
  Tag,
  TagResult,
  Reset
))

mutable struct Message
    body_type::Int8
    body::MessageBody
end

FlatBuffers.root_type(::Type) where {T<:Message} = true
FlatBuffers.file_identifier(::Type) where {T<:Message} = "PPXF"

Message(buf::AbstractVector{UInt8}) = FlatBuffers.read(Message, buf)
Message(io::IO) = FlatBuffers.deserialize(io, Message)