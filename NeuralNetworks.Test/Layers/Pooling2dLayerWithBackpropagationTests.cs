using AwesomeAssertions;
using Ivankarez.NeuralNetworks.Layers;
using Ivankarez.NeuralNetworks.Utils;
using NUnit.Framework;
using System;

namespace Ivankarez.NeuralNetworks.Test.Layers;

public class Pooling2dLayerWithBackpropagationTests
{
    private readonly Size2D inputSize = new Size2D(4, 4);
    private readonly Size2D windowSize = new Size2D(2, 2);
    private readonly Stride2D stride = new Stride2D(2, 2);

    [TestCase(PoolingType.Max)]
    [TestCase(PoolingType.Min)]
    [TestCase(PoolingType.Average)]
    [TestCase(PoolingType.Sum)]
    public void Update_CorrectOutputShape(PoolingType type)
    {
        var layer = new Pooling2dLayerWithBackpropagation(windowSize, stride, type);
        layer.Build(inputSize);

        var input = new float[inputSize.TotalSize];
        for (int i = 0; i < input.Length; i++) input[i] = i + 1;

        var output = layer.Update(input);
        output.Length.Should().Be(layer.OutputSize.TotalSize);
    }

    [Test]
    public void Backward_MaxPooling_PropagatesErrorToMax()
    {
        var layer = new Pooling2dLayerWithBackpropagation(windowSize, stride, PoolingType.Max);
        layer.Build(inputSize);

        var input = new float[]
        {
                1, 3, 2, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        };
        layer.Update(input);

        var error = new float[layer.OutputSize.TotalSize];
        for (int i = 0; i < error.Length; i++) error[i] = 1f;

        var inputError = layer.Backward(error, 0);

        // Nur die Maximalwerte in jedem Fenster erhalten Fehler 1, andere 0
        for (int nodeX = 0; nodeX < layer.OutputSize[0]; nodeX++)
        {
            for (int nodeY = 0; nodeY < layer.OutputSize[1]; nodeY++)
            {
                int nodeIndex = nodeX * layer.OutputSize[1] + nodeY;
                int startX = nodeX * stride.Horizontal;
                int startY = nodeY * stride.Vertical;

                // Berechne wer das Maximum im Fenster ist
                float maxVal = float.NegativeInfinity;
                int maxFlatIndex = -1;
                for (int fx = 0; fx < windowSize.Width; fx++)
                {
                    for (int fy = 0; fy < windowSize.Height; fy++)
                    {
                        int ix = startX + fx;
                        int iy = startY + fy;
                        int flatIndex = ix * inputSize.Width + iy;
                        if (input[flatIndex] > maxVal)
                        {
                            maxVal = input[flatIndex];
                            maxFlatIndex = flatIndex;
                        }
                    }
                }

                for (int fx = 0; fx < windowSize.Width; fx++)
                {
                    for (int fy = 0; fy < windowSize.Height; fy++)
                    {
                        int ix = startX + fx;
                        int iy = startY + fy;
                        int flatIndex = ix * inputSize.Width + iy;
                        if (flatIndex == maxFlatIndex)
                            inputError[flatIndex].Should().BeApproximately(1f, 1e-6f);
                        else
                            inputError[flatIndex].Should().BeApproximately(0f, 1e-6f);
                    }
                }
            }
        }
    }

    [Test]
    public void Backward_MinPooling_PropagatesErrorToMin()
    {
        var layer = new Pooling2dLayerWithBackpropagation(windowSize, stride, PoolingType.Min);
        layer.Build(inputSize);

        var input = new float[]
        {
                1, -3, 2, 4,
                -5, 6, 7, 8,
                9, 10, -11, 12,
                13, 14, 15, 16
        };
        layer.Update(input);

        var error = new float[layer.OutputSize.TotalSize];
        for (int i = 0; i < error.Length; i++) error[i] = 1f;

        var inputError = layer.Backward(error, 0);

        // Nur die Minimalwerte in jedem Fenster erhalten Fehler 1, andere 0
        for (int nodeX = 0; nodeX < layer.OutputSize[0]; nodeX++)
        {
            for (int nodeY = 0; nodeY < layer.OutputSize[1]; nodeY++)
            {
                int nodeIndex = nodeX * layer.OutputSize[1] + nodeY;
                int startX = nodeX * stride.Horizontal;
                int startY = nodeY * stride.Vertical;

                // Berechne wer das Minimum im Fenster ist
                float minVal = float.PositiveInfinity;
                int minFlatIndex = -1;
                for (int fx = 0; fx < windowSize.Width; fx++)
                {
                    for (int fy = 0; fy < windowSize.Height; fy++)
                    {
                        int ix = startX + fx;
                        int iy = startY + fy;
                        int flatIndex = ix * inputSize.Width + iy;
                        if (input[flatIndex] < minVal)
                        {
                            minVal = input[flatIndex];
                            minFlatIndex = flatIndex;
                        }
                    }
                }

                for (int fx = 0; fx < windowSize.Width; fx++)
                {
                    for (int fy = 0; fy < windowSize.Height; fy++)
                    {
                        int ix = startX + fx;
                        int iy = startY + fy;
                        int flatIndex = ix * inputSize.Width + iy;
                        if (flatIndex == minFlatIndex)
                            inputError[flatIndex].Should().BeApproximately(1f, 1e-6f);
                        else
                            inputError[flatIndex].Should().BeApproximately(0f, 1e-6f);
                    }
                }
            }
        }
    }

    [Test]
    public void Backward_AveragePooling_SplitsErrorEqually()
    {
        var layer = new Pooling2dLayerWithBackpropagation(windowSize, stride, PoolingType.Average);
        layer.Build(inputSize);

        var input = new float[inputSize.TotalSize];
        for (int i = 0; i < input.Length; i++) input[i] = i + 1;

        layer.Update(input);

        var error = new float[layer.OutputSize.TotalSize];
        for (int i = 0; i < error.Length; i++) error[i] = 2f;

        var inputError = layer.Backward(error, 0);

        for (int nodeX = 0; nodeX < layer.OutputSize[0]; nodeX++)
        {
            for (int nodeY = 0; nodeY < layer.OutputSize[1]; nodeY++)
            {
                int nodeIndex = nodeX * layer.OutputSize[1] + nodeY;
                int startX = nodeX * stride.Horizontal;
                int startY = nodeY * stride.Vertical;
                int poolArea = windowSize.Width * windowSize.Height;

                for (int fx = 0; fx < windowSize.Width; fx++)
                {
                    for (int fy = 0; fy < windowSize.Height; fy++)
                    {
                        int ix = startX + fx;
                        int iy = startY + fy;
                        int flatIndex = ix * inputSize.Width + iy;
                        inputError[flatIndex].Should().BeApproximately(error[nodeIndex] / poolArea, 1e-6f);
                    }
                }
            }
        }
    }

    [Test]
    public void Backward_SumPooling_PropagatesFullErrorToAllInWindow()
    {
        var layer = new Pooling2dLayerWithBackpropagation(windowSize, stride, PoolingType.Sum);
        layer.Build(inputSize);

        var input = new float[inputSize.TotalSize];
        for (int i = 0; i < input.Length; i++) input[i] = i + 1;

        layer.Update(input);

        var error = new float[layer.OutputSize.TotalSize];
        for (int i = 0; i < error.Length; i++) error[i] = 2f;

        var inputError = layer.Backward(error, 0);

        for (int nodeX = 0; nodeX < layer.OutputSize[0]; nodeX++)
        {
            for (int nodeY = 0; nodeY < layer.OutputSize[1]; nodeY++)
            {
                int nodeIndex = nodeX * layer.OutputSize[1] + nodeY;
                int startX = nodeX * stride.Horizontal;
                int startY = nodeY * stride.Vertical;

                for (int fx = 0; fx < windowSize.Width; fx++)
                {
                    for (int fy = 0; fy < windowSize.Height; fy++)
                    {
                        int ix = startX + fx;
                        int iy = startY + fy;
                        int flatIndex = ix * inputSize.Width + iy;
                        inputError[flatIndex].Should().BeApproximately(2f, 1e-6f);
                    }
                }
            }
        }
    }
}