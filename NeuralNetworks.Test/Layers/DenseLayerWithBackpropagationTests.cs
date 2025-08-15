using AwesomeAssertions;
using Ivankarez.NeuralNetworks.Activations;
using Ivankarez.NeuralNetworks.Api;
using Ivankarez.NeuralNetworks.Layers;
using NUnit.Framework;
using System;
using System.Collections;
using System.Collections.Generic;

namespace Ivankarez.NeuralNetworks.Test.Layers;

public class DenseLayerWithBackpropagationTests
{
    [TestCaseSource(nameof(Pattern_ShouldLearnCases1Layer))]
    public void OneLayerShouldLearnPattern(string name, float[][] inputs, float[] targets, int epochen, float learningRate)
    {
        var layer = new DenseLayerWithBackpropagation(1, new LinearActivationWithDerivat(), false, NN.Initializers.GlorotUniform(), NN.Initializers.Zeros());
        layer.Build(NN.Size.Of(2));

        var initialLoss = 0f;
        var loss = 0f;
        var lloss = 0f;

        for (int epoch = 0; epoch < epochen; epoch++)
        {
            loss = 0f;
            for (int i = 0; i < inputs.Length; i++)
            {
                var output = layer.Update(inputs[i]);
                float error = output[0] - targets[i];
                loss += error * error;

                layer.Backward([error], learningRate);
            }
            loss /= inputs.Length;
            if (epoch == 0) initialLoss = loss;
            if (epoch % (epochen / 20) == 0)
            {
                Console.WriteLine($"Epoch {epoch} - Loss: {loss:F6}");
                if (lloss > 0 && lloss < loss)
                {
                    Console.WriteLine($"Warning: Loss increased from {lloss:F6} to {loss:F6}, indicating potential overfitting or instability.");
                }
                if (lloss == loss)
                {
                    Console.WriteLine($"Convergence reached at epoch {epoch} with loss {loss:F6}.");
                    epochen = epoch;
                    break;
                }
                lloss = loss;
            }
        }

        // Nach dem Training: Ausgabe des Resultats
        Console.WriteLine($"Output nach Training ({epochen} Epochen) für '{name}': Loss: {loss:F6}");
        foreach (var input in inputs)
        {
            var output = layer.Update(input);
            foreach (var val in input)
                Console.Write($"{val:F0} ");
            var soll = targets[Array.IndexOf(inputs, input)];
            Console.WriteLine($": {output[0]:F2} ({soll:F1} {MathF.Abs(output[0] - soll) < 0.5f})");
        }

        initialLoss.Should().BeGreaterThan(loss, " der anfängliche Fehler ist wahrscheinlich groß und am Ende sollte der Fehler klein sein");
        loss.Should().BeLessThan(0.2f, " der Fehler sollte am Ende kleiner als 0.2 sein");

        foreach (var input in inputs)
        {
            var output = layer.Update(input);
            output.Should().ContainSingle(" die Ausgabe sollte nur einen Wert haben");
            output[0].Should().BeApproximately(targets[Array.IndexOf(inputs, input)], 0.5f, " die Ausgabe sollte dem Zielwert nahekommen");
        }
    }

    [TestCaseSource(nameof(Pattern_ShouldLearnCasesMultiLayer))]
    public void MultiLayerShouldLearnPattern(string name, int anzLayer, float[][] inputs, float[] targets, int epochen, float learningRate)
    {
        var layer = new DenseLayerWithBackpropagation[anzLayer];
        int lastNodeCount = inputs[0].Length;
        for (int i = 0; i < anzLayer; i++)
        {
            // Jede Schicht hat so viele Knoten wie der Input, außer die letzte, die hat 1 Knoten
            int nodeCount = (i == anzLayer - 1) ? 1 : inputs[0].Length;
            layer[i] = new DenseLayerWithBackpropagation(nodeCount, new SigmoidActivationWithDerivat(), false, NN.Initializers.GlorotUniform(new TestRandomProvider(21)), NN.Initializers.Zeros());
            layer[i].Build(NN.Size.Of(inputs[0].Length));
        }

        var initialLoss = 0f;
        var loss = 0f;
        var lloss = 1f;

        for (int epoch = 0; epoch < epochen; epoch++)
        {
            loss = 0f;
            for (int i = 0; i < inputs.Length; i++)
            {
                // Durch alle Schichten iterieren
                var value = inputs[i];
                foreach (var l in layer)
                {
                    value = l.Update(value);
                }
                float error = value[0] - targets[i];
                loss += error * error;

                // Fehler für die letzte Schicht
                float[] errorArray = [error];
                // Backward durch alle Schichten
                for (int j = layer.Length - 1; j >= 0; j--)
                {
                    errorArray = layer[j].Backward(errorArray, learningRate);
                }
            }
            loss /= inputs.Length;
            if (epoch == 0) initialLoss = loss;
            if (epoch % (epochen / 20) == 0)
            {
                Console.WriteLine($"Epoch {epoch} - Loss: {loss:F6}");
                if (lloss > 0 && lloss < loss)
                {
                    Console.WriteLine($"Warning: Loss increased from {lloss:F6} to {loss:F6}, indicating potential overfitting or instability.");
                }
                if (lloss == loss)
                {
                    Console.WriteLine($"Convergence reached at epoch {epoch} with loss {loss:F6}.");
                    epochen = epoch;
                    break;
                }
                lloss = loss;
            }
        }

        // Nach dem Training: Ausgabe des Resultats
        Console.WriteLine($"Output nach Training ({epochen} Epochen) für '{name}': Loss: {loss:F6}");
        foreach (var input in inputs)
        {
            var value = input;
            foreach (var l in layer)
            {
                value = l.Update(value);
            }
            foreach (var val in input)
            { Console.Write($"{val:F0} "); }
            var soll = targets[Array.IndexOf(inputs, input)];
            Console.WriteLine($": {value[0]:F2} ({soll:F1} {MathF.Abs(value[0] - soll) < 0.5f})");
        }

        initialLoss.Should().BeGreaterThan(loss, " der anfängliche Fehler ist wahrscheinlich groß und am Ende sollte der Fehler klein sein");
        loss.Should().BeLessThan(0.05f, " der Fehler sollte am Ende kleiner als 0.05 sein");

        foreach (var input in inputs)
        {
            var value = input;
            foreach (var l in layer)
            {
                value = l.Update(value);
            }
            value.Should().ContainSingle(" die Ausgabe sollte nur einen Wert haben");
            value[0].Should().BeApproximately(targets[Array.IndexOf(inputs, input)], 0.5f, " die Ausgabe sollte dem Zielwert nahekommen");
        }
    }

    public static IEnumerable<TestCaseData> Pattern_ShouldLearnCases1Layer()
    {
        TestCaseData testcase;

        testcase = new TestCaseData("AND", new float[][] { [0, 0], [0, 1], [1, 0], [1, 1] }, new float[] { 0, 0, 0, 1 }, 20, 0.2f);
        testcase.SetName("OneLayer-AND Pattern 20");
        yield return testcase;
        testcase = new TestCaseData("AND", new float[][] { [0, 0], [0, 1], [1, 0], [1, 1] }, new float[] { 0, 0, 0, 1 }, 100000, 0.001f);
        testcase.SetName("OneLayer-AND Pattern max");
        yield return testcase;
        testcase = new TestCaseData("OR", new float[][] { [0, 0], [0, 1], [1, 0], [1, 1] }, new float[] { 0, 1, 1, 1 }, 20, 0.2f);
        testcase.SetName("OneLayer-OR Pattern 20");
        yield return testcase;
        testcase = new TestCaseData("OR", new float[][] { [0, 0], [0, 1], [1, 0], [1, 1] }, new float[] { 0, 1, 1, 1 }, 100000, 0.001f);
        testcase.SetName("OneLayer-OR Pattern max");
        yield return testcase;
    }

    public static IEnumerable<TestCaseData> Pattern_ShouldLearnCasesMultiLayer()
    {
        TestCaseData testcase;

        testcase = new TestCaseData("AND", 3, new float[][] { [0, 0], [0, 1], [1, 0], [1, 1] }, new float[] { 0, 0, 0, 1 }, 475, 0.575f);
        testcase.SetName("MultiLayer-AND Pattern min");
        yield return testcase;
        testcase = new TestCaseData("AND", 3, new float[][] { [0, 0], [0, 1], [1, 0], [1, 1] }, new float[] { 0, 0, 0, 1 }, 30000, 0.3f);
        testcase.SetName("MultiLayer-AND Pattern max");
        yield return testcase;
        testcase = new TestCaseData("OR", 3, new float[][] { [0, 0], [0, 1], [1, 0], [1, 1] }, new float[] { 0, 1, 1, 1 }, 255, 0.57f);
        testcase.SetName("MultiLayer-OR Pattern min");
        yield return testcase;
        testcase = new TestCaseData("OR", 3, new float[][] { [0, 0], [0, 1], [1, 0], [1, 1] }, new float[] { 0, 1, 1, 1 }, 20000, 0.2f);
        testcase.SetName("MultiLayer-OR Pattern max");
        yield return testcase;
        testcase = new TestCaseData("XOR", 4, new float[][] { [0, 0], [0, 1], [1, 0], [1, 1] }, new float[] { 0, 1, 1, 0 }, 7200, .51f);
        testcase.SetName("MultiLayer-XOR Pattern");
        yield return testcase;
    }
}
