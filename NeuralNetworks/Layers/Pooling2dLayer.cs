using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.Utils;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class Pooling2dLayer : IModelLayer
    {
        public ISize OutputSize { get; private set; } = default!;
        public Size2D InputSize { get; set; } = default!;
        public NamedVectors<float> Parameters { get; }
        public NamedVectors<float> State { get; }
        public Size2D WindowSize { get; }
        public Stride2D Stride { get; }
        public PoolingType PoolingType { get; }

        protected readonly Func<int, int, float[], float> pooling;
        protected float[] nodeValues = default!;
        protected int nodeValuesWidth = default!;
        protected int nodeValuesHeight = default!;

        public Pooling2dLayer(Size2D windowSize, Stride2D stride, PoolingType poolingType)
        {
            WindowSize = windowSize ?? throw new ArgumentNullException(nameof(windowSize));
            Stride = stride ?? throw new ArgumentNullException(nameof(stride));
            PoolingType = poolingType;
            pooling = GetPooling();

            Parameters = new NamedVectors<float>();
            State = new NamedVectors<float>();
        }

        public bool IsBildet { get; private set; } = false;
        public void Build(ISize inputSize)
        {
            if (inputSize is not Size2D) { throw new ArgumentException($"Input size must be {nameof(Size2D)}", nameof(inputSize)); }
            InputSize = (Size2D)inputSize;

            nodeValuesWidth = ConvolutionUtils.CalculateOutputSize(InputSize.Width, WindowSize.Width, Stride.Horizontal);
            nodeValuesHeight = ConvolutionUtils.CalculateOutputSize(InputSize.Height, WindowSize.Height, Stride.Vertical);
            OutputSize = new Size2D(nodeValuesWidth, nodeValuesHeight);
            nodeValues = new float[OutputSize.TotalSize];

            State.Add("nodeValues", nodeValues);
            IsBildet = true;
        }

        public virtual float[] Update(float[] inputValues)
        {
            if (!IsBildet) throw new InvalidOperationException("Layer must be built before updating");

            for (int nodeX = 0; nodeX < nodeValuesWidth; nodeX += 1)
            {
                for (int nodeY = 0; nodeY < nodeValuesHeight; nodeY += 1)
                {
                    var nodeValue = pooling(nodeX, nodeY, inputValues);
                    var nodeIndex = nodeX * nodeValuesHeight + nodeY;
                    nodeValues[nodeIndex] = nodeValue;
                }
            }

            return nodeValues;
        }

        private Func<int, int, float[], float> GetPooling()
        {
            return PoolingType switch
            {
                PoolingType.Max => PoolByMax,
                PoolingType.Average => PoolByAverage,
                PoolingType.Min => PoolByMin,
                PoolingType.Sum => PoolBySum,
                _ => throw new NotImplementedException($"Unknown pooling {PoolingType}"),
            };
        }

        private float PoolByMax(int nodeX, int nodeY, float[] inputValues)
        {
            var nodeValue = float.NegativeInfinity;
            for (int fx = 0; fx < WindowSize.Width; fx += 1)
            {
                for (int fy = 0; fy < WindowSize.Height; fy += 1)
                {
                    var inputX = nodeX * Stride.Horizontal + fx;
                    var inputY = nodeY * Stride.Vertical + fy;
                    var inputValue = inputValues[inputX * InputSize.Width + inputY];
                    if (inputValue > nodeValue)
                    {
                        nodeValue = inputValue;
                    }
                }
            }

            return nodeValue;
        }

        private float PoolByMin(int nodeX, int nodeY, float[] inputValues)
        {
            var nodeValue = float.PositiveInfinity;
            for (int fx = 0; fx < WindowSize.Width; fx += 1)
            {
                for (int fy = 0; fy < WindowSize.Height; fy += 1)
                {
                    var inputX = nodeX * Stride.Horizontal + fx;
                    var inputY = nodeY * Stride.Vertical + fy;
                    var inputValue = inputValues[inputX * InputSize.Width + inputY];
                    if (inputValue < nodeValue)
                    {
                        nodeValue = inputValue;
                    }
                }
            }

            return nodeValue;
        }

        private float PoolBySum(int nodeX, int nodeY, float[] inputValues)
        {
            var nodeValue = 0f;
            for (int fx = 0; fx < WindowSize.Width; fx += 1)
            {
                for (int fy = 0; fy < WindowSize.Height; fy += 1)
                {
                    var inputX = nodeX * Stride.Horizontal + fx;
                    var inputY = nodeY * Stride.Vertical + fy;
                    var inputValue = inputValues[inputX * InputSize.Width + inputY];
                    nodeValue += inputValue;
                }
            }

            return nodeValue;
        }

        private float PoolByAverage(int nodeX, int nodeY, float[] inputValues)
        {
            return PoolBySum(nodeX, nodeY, inputValues) / (WindowSize.Width * WindowSize.Height);
        }
    }
}
