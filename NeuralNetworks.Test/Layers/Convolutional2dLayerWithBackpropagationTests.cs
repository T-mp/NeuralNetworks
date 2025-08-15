using AwesomeAssertions;
using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.Layers;
using Ivankarez.NeuralNetworks.RandomGeneration.Initializers;
using Ivankarez.NeuralNetworks.Utils;
using NUnit.Framework;
using System;
using System.Linq;

namespace Ivankarez.NeuralNetworks.Test.Layers;

[TestFixture]
public class Convolutional2dLayerWithBackpropagationTests
{
    private Convolutional2dLayerWithBackpropagation layer;
    private Size2D inputSize;
    private Size2D filterSize;
    private Stride2D stride;
    private float[] input;
    private IInitializer kernelInitializer;
    private IInitializer biasInitializer;
    private bool useBias;

    [SetUp]
    public void SetUp()
    {
        inputSize = new Size2D(4, 4);
        filterSize = new Size2D(2, 2);
        stride = new Stride2D(1, 1);
        useBias = true;

        // Initializer, z.B. Konstante Werte oder kleiner Zufallsfaktor
        kernelInitializer = new ConstantInitializer(1.0f); // Stellt planbare Testwerte sicher
        biasInitializer = new ConstantInitializer(0.5f);

        layer = new Convolutional2dLayerWithBackpropagation(
            filterSize, stride, useBias,
            kernelInitializer, biasInitializer
        );

        layer.Build(inputSize);

        // Input: Einfaches Raster (z.B. nur Werte 1, 2, ... 16)
        input = new float[inputSize.TotalSize];
        for (int i = 0; i < input.Length; i++)
            input[i] = i + 1;
    }

    [Test]
    public void Update_ShouldComputeConvolutionCorrectly()
    {
        var output = layer.Update(input);

        output.Length.Should().Be(layer.OutputSize.TotalSize);

        // Erwartete Werte (manuell berechnet, da Kernel=1 und Bias=0.5): jeder Output = Summe der jeweiligen 2x2-Region + 0.5
        // Hier zum Beispiel: Output[0] = input[0]+input[1]+input[4]+input[5]+0.5
        var expectedFirst =
            input[0] + input[1] + input[4] + input[5] + 0.5f;

        output[0].Should().BeApproximately(expectedFirst, 1e-5f);
    }

    [Test]
    public void Backward_ShouldPropagateErrorAndUpdateKernel()
    {
        var output = layer.Update(input);
        var outputError = new float[output.Length];
        for (int i = 0; i < outputError.Length; i++)
            outputError[i] = 1f;

        var filterBefore = layer.Parameters.Get2dVectorCopy("filter");
        var biasesBefore = useBias ? layer.Parameters.Get1dVector("biases")?.ToArray() : null;

        var inputError = layer.Backward(outputError, learningRate: 0.1f);

        // Alle Komponenten sollten berechnet werden:
        inputError.Length.Should().Be(input.Length);

        // Kernel-Update: Werte müssen sich geändert haben!
        var filterAfter = (float[,])layer.Parameters.Get2dVector("filter");
        for (int fx = 0; fx < filterAfter.GetLength(0); fx++)
            for (int fy = 0; fy < filterAfter.GetLength(1); fy++)
                filterAfter[fx, fy].Should().NotBe(filterBefore[fx, fy]);

        // Bias-Update: Werte müssen sich geändert haben!
        if (useBias && biasesBefore != null)
        {
            var biasesAfter = (float[])layer.Parameters.Get1dVector("biases");
            for (int b = 0; b < biasesAfter.Length; b++)
                biasesAfter[b].Should().NotBe(biasesBefore[b]);
        }

        // Input-Error: Es sollten Fehler rückpropagiert werden.
        inputError.Should().Contain(x => Math.Abs(x) > 0f);
    }

    [Test]
    public void Update_WithoutBuild_ShouldThrowException()
    {
        var unbuilt = new Convolutional2dLayerWithBackpropagation(
            filterSize, stride, useBias,
            kernelInitializer, biasInitializer
        );
        Assert.Throws<InvalidOperationException>(() =>
            unbuilt.Update(input));
    }

    [Test]
    public void Backward_WithoutBuild_ShouldThrowException()
    {
        var unbuilt = new Convolutional2dLayerWithBackpropagation(
            filterSize, stride, useBias,
            kernelInitializer, biasInitializer
        );
        Assert.Throws<InvalidOperationException>(() =>
            unbuilt.Backward(new float[4], 0.1f));
    }

    float[,] Copy2dArray(float[,] source)
    {
        int dim0 = source.GetLength(0);
        int dim1 = source.GetLength(1);
        var copy = new float[dim0, dim1];

        for (int i = 0; i < dim0; i++)
            for (int j = 0; j < dim1; j++)
                copy[i, j] = source[i, j];

        return copy;
    }
}