using AwesomeAssertions;
using Ivankarez.NeuralNetworks.Layers;
using Ivankarez.NeuralNetworks.Utils;
using NUnit.Framework;
using System;

namespace Ivankarez.NeuralNetworks.Test.Layers;

public class PoolingLayerWithBackpropagationTests
{
    [TestCase(PoolingType.Max)]
    [TestCase(PoolingType.Average)]
    [TestCase(PoolingType.Min)]
    [TestCase(PoolingType.Sum)]
    public void Update_CorrectOutputShape(PoolingType type)
    {
        var layer = new PoolingLayerWithBackpropagation(2, 2, type);
        var input = new float[] { 1, 3, 2, 4, 5 };
        layer.Build(new Size1D(input.Length));

        var output = layer.Update(input);
        output.Length.Should().Be(layer.OutputSize.TotalSize);
    }

    [Test]
    public void Backward_MaxPooling_PropagatesErrorToMax()
    {
        var layer = new PoolingLayerWithBackpropagation(2, 2, PoolingType.Max);
        var input = new float[] { 1, 3, 2, 4, 5, 2 };
        layer.Build(new Size1D(input.Length));
        layer.Update(input);

        var error = new float[layer.OutputSize.TotalSize];
        for (int i = 0; i < error.Length; i++) error[i] = 1f;

        var inputError = layer.Backward(error, 0);
        inputError.Should().Contain(e => e == 1f);

        // Fehler sollte exakt auf den Max-Wert pro Fenster sein
        for (int win = 0; win < layer.OutputSize.TotalSize; win++)
        {
            int start = win * layer.Stride;
            int end = Math.Min(start + layer.Window, input.Length);
            int maxIdx = start;
            float max = float.NegativeInfinity;
            for (int i = start; i < end; i++)
            {
                if (input[i] > max)
                {
                    max = input[i];
                    maxIdx = i;
                }
            }
            inputError[maxIdx].Should().BeGreaterThan(0f);
        }
    }

    [Test]
    public void Backward_MinPooling_PropagatesErrorToMax()
    {
        var layer = new PoolingLayerWithBackpropagation(2, 2, PoolingType.Min);
        var input = new float[] { 1, 3, 2, 4, 5, 2 };
        layer.Build(new Size1D(input.Length));
        layer.Update(input);

        var error = new float[layer.OutputSize.TotalSize];
        for (int i = 0; i < error.Length; i++) error[i] = 1f;

        var inputError = layer.Backward(error, 0);
        inputError.Should().Contain(e => e == 1f);

        // Fehler sollte exakt auf den Min-Wert pro Fenster sein
        for (int win = 0; win < layer.OutputSize.TotalSize; win++)
        {
            int start = win * layer.Stride;
            int windowEnd = Math.Min(start + layer.Window, input.Length);

            var min = float.PositiveInfinity;
            int minIdx = start;
            for (int i = start; i < windowEnd; i++)
            {
                var value = input[i];
                if (value < min)
                {
                    min = value;
                    minIdx = i;
                }
            }
            inputError[minIdx].Should().BeGreaterThan(0f);
        }
    }

    [Test]
    public void Backward_AveragePooling_SplitsErrorEqually()
    {
        var layer = new PoolingLayerWithBackpropagation(2, 2, PoolingType.Average);
        var input = new float[] { 1, 2, 3, 4 };
        layer.Build(new Size1D(input.Length));
        layer.Update(input);

        var error = new float[layer.OutputSize.TotalSize];
        for (int i = 0; i < error.Length; i++) error[i] = 2f;

        var inputError = layer.Backward(error, 0);
        // Jeder Wert bekommt hälftig was ab
        for (int win = 0; win < layer.OutputSize.TotalSize; win++)
        {
            int start = win * layer.Stride;
            int end = Math.Min(start + layer.Window, input.Length);
            for (int i = start; i < end; i++)
            {
                inputError[i].Should().BeApproximately(1f, 1e-6f);
            }
        }
    }

    [Test]
    public void Backward_SumPooling_PropagatesFullErrorToAllInWindow()
    {
        var layer = new PoolingLayerWithBackpropagation(window: 2, stride: 2, type: PoolingType.Sum);
        var input = new float[] { 1, 2, 3, 4 };
        layer.Build(new Size1D(input.Length));
        layer.Update(input);

        var error = new float[layer.OutputSize.TotalSize];
        for (int i = 0; i < error.Length; i++) error[i] = 2f;

        var inputError = layer.Backward(error, 0);

        // Jeder Wert in jedem Fenster erhält den vollen Fehlerwert 2f
        for (int win = 0; win < layer.OutputSize.TotalSize; win++)
        {
            int start = win * layer.Stride;
            int end = Math.Min(start + layer.Window, input.Length);
            for (int i = start; i < end; i++)
            {
                inputError[i].Should().BeApproximately(2f, 1e-6f);
            }
        }
    }

}