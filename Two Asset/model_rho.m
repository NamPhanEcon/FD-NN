function V = model_rho(parameters,az,avg_phi_K)

IJ = size(az,3);

test = repmat(avg_phi_K,[1 1 IJ]);
test3 = [az;test];

%% Evaluate rho first 

        numLayers = numel(fieldnames(parameters.rho_emp_high)) ;
        weights = parameters.rho_emp_high.fc1.Weights;
        bias = parameters.rho_emp_high.fc1.Bias;   
        V = fullyconnect(test3,weights,bias);
        % fully connect operations for remaining layers.
        for i=2:numLayers-1
            name = "fc" + i; %layer
        %   V = sigmoid(V); %need this for stable
           V = tanh(V);
         %  V = relu(V); 
            weights = parameters.rho_emp_high.(name).Weights;
            bias = parameters.rho_emp_high.(name).Bias;
            V = fullyconnect(V, weights, bias);
        end

        % Final Layer
        V = tanh(V); 
        name = "fc" + numLayers;
        weights = parameters.rho_emp_high.(name).Weights;
        bias = parameters.rho_emp_high.(name).Bias;
        V = fullyconnect(V,weights,bias); 

    end