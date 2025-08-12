using AwesomeAssertions;
using Ivankarez.NeuralNetworks.Api;
using Ivankarez.NeuralNetworks.Layers;
using NUnit.Framework;
using System;
using System.Linq;

namespace Ivankarez.NeuralNetworks.Test.Layers
{
    public class ConvolutionalLayerWithBackpropagationTests
    {
        [Test]
        public void TrainSimpleConvolutionalLayer()
        {
            // Beispielhafte Input-Daten (1D Signal)
            float[] input = new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f };

            // Ziel-Ausgabe (für Demo, z.B. eine leichte Verschiebung der Eingabe)
            float[] target = new float[] { 2f, 3f, 4f, 5f, 6f };

            // Lernrate
            float learningRate = 0.001f;//0.01f;

            // ConvolutionalLayer initialisieren (FilterSize = 4, Stride = 1, UseBias = true)
            var convLayer = new ConvolutionalLayerWithBackpropagation(filterSize: 4, stride: 1, useBias: true, NN.Initializers.GlorotUniform(), NN.Initializers.Zeros());
            convLayer.Build(NN.Size.Of(input.Length));

            float loss = 0f;

            // Trainingsloop
            for (int epoch = 0; epoch < 1000; epoch++)
            {
                // Forward-Propagation
                float[] output = convLayer.Update(input);

                // Fehler und Gradienten berechnen (Mean Squared Error Gradient)
                float[] outputError = new float[output.Length];
                for (int i = 0; i < output.Length; i++)
                {
                    outputError[i] = output[i] - target[i];
                    loss += outputError[i] * outputError[i];
                }
                loss /= output.Length;

                // Backpropagation (Gewichte anpassen)
                float[] inputError = convLayer.Backward(outputError, learningRate);

                // Optional: Ausgabe des Verlusts zur Überwachung
                if (epoch % 100 == 0)
                    Console.WriteLine($"Epoch {epoch} - Loss: {loss:F6}");
            }

            // Nach dem Training: Ausgabe des Resultats
            Console.WriteLine("Output nach Training:");
            float[] finalOutput = convLayer.Update(input);
            foreach (var val in finalOutput)
                Console.Write($"{val:F3} ");

            loss.Should().BeLessThan(0.001f);

            finalOutput.Should().HaveCount(target.Length);
            finalOutput.Should().BeEquivalentTo(target, opt => opt
                .Using<float>(ctx => ctx.Subject.Should().BeInRange(ctx.Expectation - 0.05f, ctx.Expectation + 0.05f))
                .WhenTypeIs<float>());
            Console.WriteLine();
        }

        [Test]
        public void ForwardBackward_ShouldReduceLoss_OnSimplePattern()
        {
            var input = new float[] { 1f, 2f, 3f, 4f, 5f };
            var target = new float[] { 10f, 20f, 30f };

            var layer = new ConvolutionalLayerWithBackpropagation(filterSize: 3, stride: 1, useBias: true, NN.Initializers.GlorotUniform(), NN.Initializers.Zeros());
            layer.Build(NN.Size.Of(input.Length));

            float initialLoss = 0f;

            for (int epoch = 0; epoch < 50; epoch++)
            {
                var output = layer.Update(input);

                float[] error = new float[output.Length];
                float loss = 0f;
                for (int i = 0; i < output.Length; i++)
                {
                    error[i] = output[i] - target[i];
                    loss += error[i] * error[i];
                }
                loss /= output.Length;
                if (epoch == 0) initialLoss = loss;

                layer.Backward(error, learningRate: 0.001f);
            }

            var finalOutput = layer.Update(input);
            float finalLoss = 0f;
            for (int i = 0; i < finalOutput.Length; i++)
                finalLoss += (finalOutput[i] - target[i]) * (finalOutput[i] - target[i]);
            finalLoss /= finalOutput.Length;

            finalLoss.Should().BeLessThan(initialLoss);
        }

        [Test]
        public void Backward_ShouldNotAffectOutput_WhenUseBiasFalse()
        {
            var input = new float[] { 0f, 0f, 0f };
            // Layer ohne Bias
            var layerNoBias = new ConvolutionalLayerWithBackpropagation(filterSize: 2, stride: 1, useBias: false, NN.Initializers.Constant(0.5f), NN.Initializers.Constant(0.5f));
            layerNoBias.Build(NN.Size.Of(input.Length));

            var originalOutput = layerNoBias.Update(input).ToArray();

            layerNoBias.Backward(new float[] { 1f, 1f }, learningRate: 0.1f);

            var newOutput = layerNoBias.Update(input).ToArray();

            newOutput.Should().BeEquivalentTo(originalOutput);
        }

        [Test]
        public void Backward_ShouldAffectOutput_WhenUseBias()
        {
            var input = new float[] { 0f, 0f, 0f };
            // Layer ohne Bias
            var layerNoBias = new ConvolutionalLayerWithBackpropagation(
                filterSize: 2, 
                stride: 1, 
                useBias: true, 
                NN.Initializers.Constant(1.0f), 
                NN.Initializers.Constant(0.5f));
            layerNoBias.Build(NN.Size.Of(input.Length));

            var originalOutput = layerNoBias.Update(input).ToArray();

            originalOutput.Should().BeEquivalentTo([0.5f, 0.5f]); //0 + Bias

            float[] floats = layerNoBias.Backward([ 1f, 1f ], learningRate: 0.1f);

            var newOutput = layerNoBias.Update(input).ToArray();

            newOutput.Should().BeEquivalentTo([0.4f, 0.4f]); //0 + Bias (-learningRate)

            newOutput.Should().NotBeEquivalentTo(originalOutput);
        }

    }
}
