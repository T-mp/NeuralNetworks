using Ivankarez.NeuralNetworks.Abstractions;
using System;

namespace Ivankarez.NeuralNetworks.Test.Layers;

// Minimale deterministische Test-Implementierung des Zufallsproviders.
// Falls euer IRandomProvider andere Signaturen hat, bitte die Methoden anpassen.
internal sealed class TestRandomProvider : IRandomProvider
{
    private readonly Random rng;

    public TestRandomProvider(int seed)
    {
        rng = new Random(seed);
    }

    public float NextFloat()
    {
        return (float)rng.NextDouble();
    }
}