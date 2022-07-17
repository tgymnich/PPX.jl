using FlatBuffers
using PPX
using Test

@testset "Test reading of a Tensor" begin
    data = read("tensor.bin")
    tensor = PPX.Tensor(data)

    @test tensor.data == [1.0, 2.0, 3.0, 4.0]
    @test tensor.shape == [4]
end

@testset "Test reading of a Normal Distribution" begin
    data = read("normal.bin")
    normal = PPX.Normal(data)

    @test normal.mean.data == [0.0]
    @test normal.mean.shape == [1]
    @test normal.stddev.data == [1.0]
    @test normal.stddev.shape == [1]
end

@testset "Test reading of a Sample" begin
    data = read("sample.bin")
    sample = PPX.Sample(data)

    @test sample.address == "SampleAddress"
    @test sample.name == "SampleName"
    @test sample.control == false
    @test sample.distribution.mean.data == [0.0]
    @test sample.distribution.mean.shape == [1]
    @test sample.distribution.stddev.data == [1.0]
    @test sample.distribution.stddev.shape == [1]
end

@testset "Test reading of a Message" begin
    data = read("message.bin")
    msg = PPX.Message(data)

    @test msg.body.address == "SampleAddress"
    @test msg.body.name == "SampleName"
    @test msg.body.control == false
    @test msg.body.distribution.mean.data == [0.0]
    @test msg.body.distribution.mean.shape == [1]
    @test msg.body.distribution.stddev.data == [1.0]
    @test msg.body.distribution.stddev.shape == [1]
end