namespace Ivankarez.NeuralNetworks.Abstractions
{
    public interface IModelLayerWithBackpropagation : IModelLayer
    {
        /// <summary>
        /// Backpropagation für die Schicht
        /// </summary>
        /// <param name="error">Der Fehler, der von der nächsten Schicht zurückgegeben wurde</param>
        /// <param name="learningRate">Die Lernrate für die Anpassung der Gewichte</param>
        float[] Backward(float[] error, float learningRate);
    }
}
