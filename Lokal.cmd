dotnet restore
dotnet build --no-restore -c Release
dotnet test --no-restore --no-build -c Release --verbosity normal --results-directory ./testresults
dotnet pack NeuralNetworks --no-build --no-restore -c Release -o packages -p:RepositoryUrl=https://github.com/T-mp/NeuralNetworks