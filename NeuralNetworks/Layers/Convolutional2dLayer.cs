using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.RandomGeneration;
using Ivankarez.NeuralNetworks.Utils;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers;

public class Convolutional2dLayer : IModelLayer
{
    public ISize OutputSize { get; private set; } = default!;
    public NamedVectors<float> Parameters { get; }
    public NamedVectors<float> State { get; }
    public Size2D InputSize { get; set; } = default!;
    public Size2D FilterSize { get; }
    public Stride2D Stride { get; }
    public bool UseBias { get; }
    public IInitializer KernelInitializer { get; }
    public IInitializer BiasInitializer { get; }

    public bool IsBildet { get; private set; }

    protected float[] nodeValues = default!;
    protected float[] biases = default!;
    protected float[,] filter = default!;
    protected int outputWidth = default!;
    protected int outputHeight = default!;

    public Convolutional2dLayer(Size2D filterSize, Stride2D stride,
        bool useBias, IInitializer kernelInitializer, IInitializer biasInitializer)
    {
        Parameters = new NamedVectors<float>();
        State = new NamedVectors<float>();
        FilterSize = filterSize ?? throw new ArgumentNullException(nameof(filterSize));
        Stride = stride ?? throw new ArgumentNullException(nameof(stride));
        UseBias = useBias;
        KernelInitializer = kernelInitializer ?? throw new ArgumentNullException(nameof(kernelInitializer));
        BiasInitializer = biasInitializer ?? throw new ArgumentNullException(nameof(biasInitializer));
    }

    public void Build(ISize inputSize)
    {
        IsBildet = true;
        if (inputSize is not Size2D)
        {
            throw new ArgumentException($"Input size must be {nameof(Size2D)}", nameof(inputSize));
        }
        InputSize = (Size2D)inputSize;
        outputWidth = ConvolutionUtils.CalculateOutputSize(InputSize.Width, FilterSize.Width, Stride.Horizontal);
        outputHeight = ConvolutionUtils.CalculateOutputSize(InputSize.Height, FilterSize.Height, Stride.Vertical);
        OutputSize = new Size2D(outputWidth, outputHeight);
        nodeValues = new float[OutputSize.TotalSize];
        filter = KernelInitializer.GenerateValues2d(inputSize.TotalSize, OutputSize.TotalSize, FilterSize.Width, FilterSize.Height);
        if (UseBias)
        {
            biases = BiasInitializer.GenerateValues(inputSize.TotalSize, OutputSize.TotalSize, OutputSize.TotalSize);
            Parameters.Add("biases", biases);
        }

        State.Add("nodeValues", nodeValues);
        Parameters.Add("filter", filter);
    }

    public virtual float[] Update(float[] inputValues)
    {
        if (!IsBildet)
        {
            throw new InvalidOperationException("Layer must be built before updating.");
        }
        for (int nodeX = 0; nodeX < outputWidth; nodeX += 1)
        {
            for (int nodeY = 0; nodeY < outputHeight; nodeY += 1)
            {
                var nodeValue = 0f;
                for (int fx = 0; fx < filter.GetLength(0); fx += 1)
                {
                    for (int fy = 0; fy < filter.GetLength(1); fy += 1)
                    {
                        var inputX = nodeX * Stride.Horizontal + fx;
                        var inputY = nodeY * Stride.Vertical + fy;
                        nodeValue += inputValues[inputX * InputSize.Width + inputY] * filter[fx, fy];
                    }
                }
                var nodeIndex = nodeX * outputHeight + nodeY;
                if (UseBias)
                {
                    nodeValue += biases[nodeIndex];
                }
                nodeValues[nodeIndex] = nodeValue;
            }
        }

        return nodeValues;
    }
}
