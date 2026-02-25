
netParamsV = struct();

%% ---------------- PHI NETWORK ----------------
netParamsV.phi = struct;

% fc1
sz = [phi_numNeurons, input_phi];
netParamsV.phi.fc1.Weights = initializeNormalizedXavier(sz);
netParamsV.phi.fc1.Bias    = initializeZeros([phi_numNeurons, 1]);

% hidden layers fc2 ... fc_{phi_numLayers-1}
for layerNumber = 2:phi_numLayers-1
    name = "fc" + layerNumber;
    sz = [phi_numNeurons, phi_numNeurons];
    netParamsV.phi.(name).Weights = initializeNormalizedXavier(sz);
    netParamsV.phi.(name).Bias    = initializeZeros([phi_numNeurons, 1]);
end

% final layer fc_{phi_numLayers}
sz = [L, phi_numNeurons];
netParamsV.phi.("fc" + phi_numLayers).Weights = initializeNormalizedXavier(sz);
netParamsV.phi.("fc" + phi_numLayers).Bias    = initializeZeros([L, 1]);

%% ---------------- RHO NETWORK ----------------

netParamsV.rho = struct;
rho_input =  L + number_individual_state_variable + number_aggregate_state_variable;

% fc1
sz = [rho_numNeurons, rho_input];
netParamsV.rho.fc1.Weights = initializeNormalizedXavier(sz);
netParamsV.rho.fc1.Bias    = initializeZeros([rho_numNeurons, 1]);

% hidden layers fc2 ... fc_{rho_numLayers-1}
for layerNumber = 2:rho_numLayers-1
    name = "fc" + layerNumber;
    sz = [rho_numNeurons, rho_numNeurons];
    netParamsV.rho.(name).Weights = initializeNormalizedXavier(sz);
    netParamsV.rho.(name).Bias    = initializeZeros([rho_numNeurons, 1]);
end

% final layer fc_{rho_numLayers}
sz = [1, rho_numNeurons];
netParamsV.rho.("fc" + rho_numLayers).Weights = initializeNormalizedXavier(sz);
netParamsV.rho.("fc" + rho_numLayers).Bias    = initializeZeros([1, 1]);