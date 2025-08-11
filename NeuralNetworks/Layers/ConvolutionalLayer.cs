﻿using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.RandomGeneration;
using Ivankarez.NeuralNetworks.Utils;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class ConvolutionalLayer : IModelLayer
    {
        public ISize OutputSize { get; private set; } = default!;
        public int FilterSize { get; }
        public int Stride { get; }
        public bool UseBias { get; }
        public IInitializer KernelInitializer { get; }
        public IInitializer BiasInitializer { get; }
        public NamedVectors<float> Parameters { get; }
        public NamedVectors<float> State { get; }

        protected float[] nodeValues = default!;
        protected float[] filter = default!;
        protected float[] biases = default!;

        public ConvolutionalLayer(int filterSize, int stride, bool useBias, IInitializer kernelInitializer, IInitializer biasInitializer)
        {
            if (filterSize < 1) throw new ArgumentException("Filter size must be greater than 0", nameof(filterSize));
            if (stride < 1) throw new ArgumentException("Stride must be greater than 0", nameof(stride));
            FilterSize = filterSize;
            Stride = stride;
            UseBias = useBias;
            KernelInitializer = kernelInitializer ?? throw new ArgumentNullException(nameof(kernelInitializer));
            BiasInitializer = biasInitializer ?? throw new ArgumentNullException(nameof(biasInitializer));
            Parameters = new NamedVectors<float>();
            State = new NamedVectors<float>();
        }

        public void Build(ISize inputSize)
        {
            if (FilterSize > inputSize.TotalSize) throw new ArgumentException("filterSize cannot be more than the size of the previous layer", nameof(inputSize));
            OutputSize = new Size1D(ConvolutionUtils.CalculateOutputSize(inputSize.TotalSize, FilterSize, Stride));

            nodeValues = new float[OutputSize.TotalSize];
            filter = KernelInitializer.GenerateValues(inputSize.TotalSize, OutputSize.TotalSize, FilterSize);
            if (UseBias)
            {
                biases = BiasInitializer.GenerateValues(OutputSize.TotalSize, OutputSize.TotalSize, OutputSize.TotalSize);
                Parameters.Add("biases", biases);
            }

            State.Add("nodeValues", nodeValues);
            Parameters.Add("filter", filter);
        }

        public virtual float[] Update(float[] inputValues)
        {
            for (int kernelIndex = 0; kernelIndex < OutputSize.TotalSize; kernelIndex++)
            {
                var value = DotProductWithFilter(inputValues, kernelIndex * Stride);
                if (UseBias)
                {
                    value += biases[kernelIndex];
                }
                nodeValues[kernelIndex] = value;
            }

            return nodeValues;
        }

        private float DotProductWithFilter(ReadOnlySpan<float> inputValue, int windowStart)
        {
            var sum = 0f;
            for (int i = 0; i < FilterSize; i++)
            {
                sum += inputValue[windowStart + i] * filter[i];
            }

            return sum;
        }
    }
}
