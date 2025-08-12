using AwesomeAssertions;
using Ivankarez.NeuralNetworks.Api;
using Ivankarez.NeuralNetworks.Layers;
using NUnit.Framework;
using System;
using System.Linq;

namespace Ivankarez.NeuralNetworks.Test.Layers;

public class DropoutLayerWithBackpropagationTests
    {
        private DropoutLayerWithBackpropagation layer;
        private float[] input;
        private int inputSize;
        private float dropoutRate;
        private TestRandomProvider rng;

        [SetUp]
        public void SetUp()
        {
            dropoutRate = 0.5f;
            input = [1f, 2f, 3f, 4f];
            inputSize = input.Length;

            rng = new TestRandomProvider(seed: 1234);

            layer = new DropoutLayerWithBackpropagation(dropoutRate, rng);
            layer.Build(NN.Size.Of(inputSize));
        }

        [Test]
        public void Update_TrainingMode_ShouldApplyMask()
        {
            var output = layer.Update(input);

            output.Count(v => v == 0f).Should().BeGreaterThan(0);
            output.Count(v => v != 0f).Should().BeGreaterThan(0);

            // Maskenlogik: aktive Einträge bleiben identisch, gedroppte werden 0
            for (int i = 0; i < output.Length; i++)
            {
                if (!layer.mask[i])
                    output[i].Should().Be(0f);
                else
                    output[i].Should().Be(input[i]);
            }
        }

        [Test]
        public void Update_WithDropoutRateZero_ShouldReturnInputUnchanged()
        {
            layer = new DropoutLayerWithBackpropagation(0f, new TestRandomProvider(42));
            layer.Build(NN.Size.Of(inputSize));

            var output = layer.Update(input);

            output.Should().BeEquivalentTo([0,0,0,0]);
        }

        [Test]
        public void Update_WithHighDropout_ShouldDropMostUnits()
        {
            layer = new DropoutLayerWithBackpropagation(0.9f, new TestRandomProvider(42));
            layer.Build(NN.Size.Of(inputSize));

            var output = layer.Update(input);

            output.Count(v => v != 0f).Should().BeGreaterThanOrEqualTo(3);
        }

        [Test]
        public void Backward_ShouldPassGradientOnlyThroughActiveMask()
        {
            layer.Update(input); // erzeugt Maske
            var upstream = new float[] { 0.1f, 0.2f, 0.3f, 0.4f };

            var back = layer.Backward(upstream, learningRate: 0.01f);

            for (int i = 0; i < back.Length; i++)
            {
                if (!layer.mask[i])
                    back[i].Should().Be(0f);
                else
                    back[i].Should().Be(upstream[i]);
            }
        }

        [Test]
        public void Build_MustBeCalledBeforeUse()
        {
            var fresh = new DropoutLayerWithBackpropagation(dropoutRate, new TestRandomProvider(99));

            Assert.Throws<InvalidOperationException>(() => fresh.Update(input));
            Assert.Throws<InvalidOperationException>(() => fresh.Backward(input, 0.1f));

            fresh.Build(NN.Size.Of(inputSize));
            var output = fresh.Update(input);
            output.Length.Should().Be(inputSize);
        }

        [Test]
        public void Update_RepeatedCalls_ShouldRegenerateMask()
        {
            // Bei erneutem Update sollte mit gleichem RNG-Status eine neue Maske entstehen.
            // Da RNG fortschreitet, erwarten wir i.d.R. eine andere Verteilung 0/!=0.
            var out1 = layer.Update(input).ToArray();
            var out2 = layer.Update(input);

            // Nicht strikt identisch; mindestens eine Abweichung
            out2.SequenceEqual(out1).Should().BeFalse();
        }
    }
