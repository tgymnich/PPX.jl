using FlatBuffers

include("Tensor.jl")
include("Utils.jl")

mutable struct Normal
    mean::Tensor
    stddev::Tensor
end

Normal(buf::AbstractVector{UInt8}) = FlatBuffers.read(Normal, buf)
Normal(io::IO) = FlatBuffers.deserialize(io, Normal)

mutable struct Uniform
    low::Tensor
    high::Tensor
end

Uniform(buf::AbstractVector{UInt8}) = FlatBuffers.read(Uniform, buf)
Uniform(io::IO) = FlatBuffers.deserialize(io, Uniform)

mutable struct Categorical
    probs::Tensor
end

Categorical(buf::AbstractVector{UInt8}) = FlatBuffers.read(Categorical, buf)
Categorical(io::IO) = FlatBuffers.deserialize(io, Categorical)

mutable struct Poisson
    rate::Tensor
end

Poisson(buf::AbstractVector{UInt8}) = FlatBuffers.read(Poisson, buf)
Poisson(io::IO) = FlatBuffers.deserialize(io, Poisson)

mutable struct Bernoulli
    probs::Tensor
end

Bernoulli(buf::AbstractVector{UInt8}) = FlatBuffers.read(Bernoulli, buf)
Bernoulli(io::IO) = FlatBuffers.deserialize(io, Bernoulli)

mutable struct Beta
    concentration1::Tensor
    concentration2::Tensor
end

Beta(buf::AbstractVector{UInt8}) = FlatBuffers.read(Beta, buf)
Beta(io::IO) = FlatBuffers.deserialize(io, Beta)

mutable struct Exponential
    rate::Tensor
end

Exponential(buf::AbstractVector{UInt8}) = FlatBuffers.read(Exponential, buf)
Exponential(io::IO) = FlatBuffers.deserialize(io, Exponential)

mutable struct Gamma
    concentration::Tensor
    rate::Tensor
end

Gamma(buf::AbstractVector{UInt8}) = FlatBuffers.read(Gamma, buf)
Gamma(io::IO) = FlatBuffers.deserialize(io, Gamma)

mutable struct LogNormal
    loc::Tensor
    scale::Tensor
end

LogNormal(buf::AbstractVector{UInt8}) = FlatBuffers.read(LogNormal, buf)
LogNormal(io::IO) = FlatBuffers.deserialize(io, LogNormal)

mutable struct Binomial
    total_count::Tensor
    probs::Tensor
end

Binomial(buf::AbstractVector{UInt8}) = FlatBuffers.read(Binomial, buf)
Binomial(io::IO) = FlatBuffers.deserialize(io, Binomial)

mutable struct Weibull
    scale::Tensor
    concentration::Tensor
end

Weibull(buf::AbstractVector{UInt8}) = FlatBuffers.read(Weibull, buf)
Weibull(io::IO) = FlatBuffers.deserialize(io, Weibull)

FlatBuffers.@UNION(Distribution, (
  NONE,
  Normal,
  Uniform,
  Categorical,
  Poisson,
  Bernoulli,
  Beta,
  Exponential,
  Gamma,
  LogNormal,
  Binomial,
  Weibull
))