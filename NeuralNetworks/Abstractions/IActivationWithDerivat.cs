namespace Ivankarez.NeuralNetworks.Abstractions
{
    public interface IActivationWithDerivat:IActivation
    {
        /// <summary>
        /// Berechnet den Werde der Ableitungsfunktion für die Aktivierung.
        /// </summary>
        /// <param name="input">der wert vor der Aktivierung</param>
        /// <returns>Das Ergebnis wenn <see cref="input"/> in die Ableitung übergeben wurde</returns>
        public float Derivat(float input);
    }
}
