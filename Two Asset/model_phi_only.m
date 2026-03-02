function [avg_phi_K] =  model_phi_only(phi,AZ)
%% Evaluate phi first 
        numLayers = numel(fieldnames(phi));
        weights = phi.fc1.Weights;
        bias = phi.fc1.Bias;
     
        V = fullyconnect(AZ,weights,bias);
        % fully connect operations for remaining layers.
        for i=2:numLayers-1
            name = "fc" + i; %layer
            %V = relu(V); 
            V = tanh(V);
            weights = phi.(name).Weights;
            bias = phi.(name).Bias;
            V = fullyconnect(V, weights, bias);
        end

        % Skip layer
        % numLayers1 = numLayers-1;
        % name = "fc" + numLayers1;
        % weights = parameters.phi.(name).Weights;
        % bias = parameters.phi.(name).Bias;
        % Vskip = fullyconnect(AZ,weights,bias);
        % V=V+Vskip;
        % 
        % Final Layer
 
     %   V = relu(V);
        V = tanh(V); 
        name = "fc" + numLayers;
        weights = phi.(name).Weights;
        bias = phi.(name).Bias;
        V = fullyconnect(V,weights,bias); 

        avg_phi_K = mean(V,3);
