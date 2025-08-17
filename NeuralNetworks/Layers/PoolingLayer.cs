using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.Utils;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers;

public class PoolingLayer : IModelLayer
{
    public ISize OutputSize { get; private set; } = default!;
    public int Window { get; }
    public int Stride { get; }
    public PoolingType Type { get; }
    public NamedVectors<float> Parameters { get; }
    public NamedVectors<float> State { get; }

    protected readonly Func<int, float[], float> pooling;
    protected float[] nodeValues = default!;

    public PoolingLayer(int window, int stride, PoolingType type)
    {
        if (window < 1) throw new ArgumentException("Window must be greater than 0", nameof(window));
        if (stride < 1) throw new ArgumentException("Stride must be greater than 0", nameof(stride));

        Window = window;
        Stride = stride;
        Type = type;

        Parameters = new NamedVectors<float>();
        State = new NamedVectors<float>();

        pooling = GetPooling();
    }

    public bool IsBildet { get; private set; } = false;
    public void Build(ISize inputSize)
    {
        OutputSize = new Size1D(ConvolutionUtils.CalculateOutputSize(inputSize.TotalSize, Window, Stride));
        nodeValues = new float[OutputSize.TotalSize];
        State.Add("nodeValues", nodeValues);

        IsBildet = true;
    }
    private Func<int, float[], float> GetPooling()
    {
        return Type switch
        {
            PoolingType.Max => PoolByMaximum,
            PoolingType.Average => PoolByAverage,
            PoolingType.Min => PoolByMinimum,
            PoolingType.Sum => PoolBySum,
            _ => throw new NotImplementedException($"Unknown pooling {Type}"),
        };
    }
    public virtual float[] Update(float[] inputValues)
    {
        if (!IsBildet) throw new InvalidOperationException("Layer must be built before updating");

        for (int nodeIndex = 0; nodeIndex < nodeValues.Length; nodeIndex++)
        {
            var startIndex = nodeIndex * Stride;
            nodeValues[nodeIndex] = pooling(startIndex, inputValues);
        }

        return nodeValues;
    }

    private float PoolByMaximum(int start, float[] inputValues)
    {
        var windowEnd = Math.Min(start + Window, inputValues.Length);
        var max = float.NegativeInfinity;
        for (int i = start; i < windowEnd; i++)
        {
            var value = inputValues[i];
            if (value > max)
            {
                max = value;
                BackpropagationStateSet("minMaxIds", start, i);
            }
        }

        return max;
    }

    private float PoolByMinimum(int start, float[] inputValues)
    {
        var windowEnd = Math.Min(start + Window, inputValues.Length);
        var min = float.PositiveInfinity;
        for (int i = start; i < windowEnd; i++)
        {
            var value = inputValues[i];
            if (value < min)
            {
                min = value;
                BackpropagationStateSet("minMaxIds", start, i);
            }
        }
        return min;
    }

    private float PoolByAverage(int start, float[] inputValues)
    {
        var windowEnd = Math.Min(start + Window, inputValues.Length);
        return PoolBySum(start, inputValues) / (windowEnd - start);
    }

    private float PoolBySum(int start, float[] inputValues)
    {
        var windowEnd = Math.Min(start + Window, inputValues.Length);
        var sum = 0f;
        for (int i = start; i < windowEnd; i++)
        {
            sum += inputValues[i];
        }
        return sum;
    }

    protected virtual void BackpropagationStateSet(string name, int index, float value) { }
}
