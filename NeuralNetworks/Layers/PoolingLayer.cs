﻿using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.Utils;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class PoolingLayer : IModelLayer
    {
        public ISize OutputSize { get; private set; } = default!;
        public int Window { get; }
        public int Stride { get; }
        public PoolingType Type { get; }
        public NamedVectors<float> Parameters { get; }
        public NamedVectors<float> State { get; }

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
        }

        public void Build(ISize inputSize)
        {
            OutputSize = new Size1D(ConvolutionUtils.CalculateOutputSize(inputSize.TotalSize, Window, Stride));
            nodeValues = new float[OutputSize.TotalSize];
            State.Add("nodeValues", nodeValues);
        }

        public virtual float[] Update(float[] inputValues)
        {
            for (int nodeIndex = 0; nodeIndex < nodeValues.Length; nodeIndex++)
            {
                var startIndex = nodeIndex * Stride;
                if (Type == PoolingType.Max)
                {
                    nodeValues[nodeIndex] = PoolByMaximum(startIndex, inputValues);
                }
                else if (Type == PoolingType.Average)
                {
                    nodeValues[nodeIndex] = PoolByAverage(startIndex, inputValues);
                }
                else if (Type == PoolingType.Min)
                {
                    nodeValues[nodeIndex] = PoolByMinimum(startIndex, inputValues);
                }
                else if (Type == PoolingType.Sum)
                {
                    nodeValues[nodeIndex] = PoolBySum(startIndex, inputValues);
                }
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
    }
}
