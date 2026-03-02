function V = model_DL_V(parameters,az,B,A,Y,Z)

IJ = size(az,3);
AZ = cat(1,B,A,Y,Z);
%% Evaluate phi first 
        numLayers = numel(fieldnames(parameters.phi));
        weights = parameters.phi.fc1.Weights;
        bias = parameters.phi.fc1.Bias;
     
        V = fullyconnect(AZ,weights,bias);
        % fully connect operations for remaining layers.
        for i=2:numLayers-1
            name = "fc" + i; %layer
            %V = relu(V); 
            V = tanh(V);
            weights = parameters.phi.(name).Weights;
            bias = parameters.phi.(name).Bias;
            V = fullyconnect(V, weights, bias);
        end

        V = tanh(V); 
        name = "fc" + numLayers;
        weights = parameters.phi.(name).Weights;
        bias = parameters.phi.(name).Bias;
        V = fullyconnect(V,weights,bias); 
        
avg_phi_K = mean(V,3);
test = repmat(avg_phi_K,[1 1 IJ]);

test3 = [az;test];

%% Evaluate rho first 

        numLayers = numel(fieldnames(parameters.rho));
        weights = parameters.rho.fc1.Weights;
        bias = parameters.rho.fc1.Bias;   
        V = fullyconnect(test3,weights,bias);
        % fully connect operations for remaining layers.
        for i=2:numLayers-1
            name = "fc" + i; %layer
        %   V = sigmoid(V); %need this for stable
           V = tanh(V);
         %  V = relu(V); 
            weights = parameters.rho.(name).Weights;
            bias = parameters.rho.(name).Bias;
            V = fullyconnect(V, weights, bias);
        end

 
        % Final Layer
        V = tanh(V); 
        name = "fc" + numLayers;
        weights = parameters.rho.(name).Weights;
        bias = parameters.rho.(name).Bias;
        V = fullyconnect(V,weights,bias); 

       % V = tanh(V); 
    end