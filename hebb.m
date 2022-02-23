
%   wiki_data - input data.
% Data input and intilization of both image and text nets

%clear
%tic

%loading of mat features into workspace
load("wikipedia_dataset\raw_features.mat");

input_I = I_tr';
input_T = T_tr';


input_I_nrows = size(input_I,1);
input_I_ncols = size(input_I,2);
input_T_nrows = size(input_T,1);
input_T_ncols = size(input_T,2);

% Create a Self-Organizing Map
dimension1 = 4;
dimension2 = 4;
net_I = selforgmap([dimension1 dimension2], 200);
net_T = selforgmap([dimension1 dimension2], 200);

%%
% Train the Network
net_I = train(net_I,input_I);
net_T = train(net_T,input_T);

% Test the Network
y_I = net_I(input_I);
y_T = net_T(input_T);

% Getting node number for each input instance
classes_I = vec2ind(y_I);
classes_T = vec2ind(y_T);

% View the Network
view(net_I)
view(net_T)

%%
% Plots
% Uncomment these lines to enable various plots.
%%figure, plotsomtop(net)
%%figure, plotsomnc(net)

%figure, plotsomnd(net_I)
%figure, plotsomhits(net_I,input_I)
%figure, plotsompos(net_I,input_I)

%figure, plotsomnd(net_T)
%figure, plotsomhits(net_T,input_T)
%figure, plotsompos(net_T,input_T)

%%figure, plotwb(net);


%%
% Getting all the node weights
nw_I = net_I.IW;
nodeWeights_I = nw_I{1,1};

nw_T = net_T.IW;
nodeWeights_T = nw_T{1,1};

%%
% weight (winners) matrix creation corresponding to input data ( for each input instance)
winnersMatrix_I = zeros(input_I_nrows,input_I_ncols);
winnersMatrix_T = zeros(input_T_nrows,input_T_ncols);
winner_I = 0; 
winner_T = 0; 
for i = 1:length(classes_I)
    winner_I = classes_I(i);
    winnersMatrix_I(:,i) = nodeWeights_I(winner_I,:)';
end

for i = 1:length(classes_T)
    winner_T = classes_T(i);
    winnersMatrix_T(:,i) = nodeWeights_T(winner_T,:)';
end

%%
% Calculating the Euclidean distance between each input instance and 
% corresponding winner node and generate one number for an instance. 
% whebb_I - image (input) instances for hebbian network
% whebb_T - text (output) instances for hebbian network
% nd_net_I - node dimension of image net
% nd_net_T - node dimension of text net
whebb_I = zeros(1,input_I_ncols);
whebb_T = zeros(1,input_T_ncols);
nd_net_I = input_I_nrows;
nd_net_T = input_T_nrows;

for i = 1:length(classes_I)
    for j = 1:nd_net_I
        whebb_I(i) = whebb_I(i) + (winnersMatrix_I(j,i) - input_I(j,i))^2;
    end
    for j = 1:nd_net_T
        whebb_T(i) = whebb_T(i) + (winnersMatrix_T(j,i) - input_T(j,i))^2;
    end
    whebb_I(i) = sqrt(whebb_I(i));
    whebb_T(i) = sqrt(whebb_T(i));
end

%%
% Training of Hebbian Network
net_I_size = dimension1*dimension2;
net_T_size = dimension1*dimension2;
hebbLink = zeros(net_I_size,net_T_size);
learningRate = 0.01;
for i=1:length(classes_I)
    hebbLink(classes_I(i),classes_T(i)) = hebbLink(classes_I(i),classes_T(i)) + learningRate * whebb_I(i) * whebb_T(i);
end

%%
% hebbLink rescaling
% hebbLink = normalize(hebbLink);

%%
% Associate input with output and output with input
% ass_netinput represents image net here
% ass_netoutput represents text net here
ass_netinput = zeros(1,net_I_size);
ass_netoutput = zeros(1,net_T_size);

% Associate output (text) to input (image)
for i=1:net_I_size
    maxtemp = 0;
    maxindex = -1;
    
    for j=1:net_T_size
        if hebbLink(i,j) > maxtemp
            maxtemp = hebbLink(i,j);
            %disp(maxtemp)
            maxindex = j;
        end
    end
    ass_netinput(i) = maxindex;
end

% Associate input (image) to output (text)
for i=1:net_T_size
    maxtemp = 0;
    maxindex = -1;
    for j=1:net_I_size
        if hebbLink(j,i) > maxtemp
            maxtemp = hebbLink(j,i);
            maxindex = j;
        end
    end
    ass_netoutput(i) = maxindex;
end

%%
% testing with image query
image_testdata = I_te';
text_testdata = T_te';


y_test = net_I(image_testdata);
classes_test = vec2ind(y_test);
image_testdata_len = length(classes_test); 

% text_output saves the corresponding text SOM node for each test instance
text_output = zeros(1,image_testdata_len);
for i = 1:image_testdata_len
    for j = 1:length(ass_netinput)
        if j == classes_test(i)
            text_output(i) = ass_netinput(j);
        end
    end
end

%%
% Fetch the results for each of the image test instance
% Or we have to fetch the text input instances which have been clustered 
% in the same node as the test image instance is linked with. 

% Initialization of AP array
AP = zeros(1,image_testdata_len);

% Starting AP calculation
% It has been calculated for each test query one by one
for i = 1:image_testdata_len
    k=1;
    % Finding index numbers of those inputs which have been clustered in
    % the same cluster as the test_outcome
    for j = 1:length(classes_T)
        if text_output(i) == classes_T(j)
            indexno(k) = j;
            k = k+1;
        end
    end
    
    
    % initialization of similar_instances and similarity.
    % similar_instances - saves all the instances or inputs which are
    % present at index numbers mentioned in indexno array.
    % similarity array saves the euclidean distance between text instance
    % and all the retrieved results.
    similar_instances = zeros(input_T_nrows,length(indexno));
    similarity = zeros(1,length(indexno));
    for j = 1:length(indexno)
        similar_instances(:,j) = input_T(:,indexno(j));
        similarity(j) = norm(text_testdata(:,i) - similar_instances(:,j)); 
    end
    % calculation of mean euclidean distance
    mean_similarity = mean(similarity);
    
    % average precision calculation for a query or test instance
    P = zeros(1,length(similarity)); % precision for each outcome of a query. 
    rel = 0;  % relevance
    for j = 1:length(similarity)
        if similarity(j) > mean_similarity   % means it is relevant
            rel = rel+1;
            P(j) = rel/j;
        end
    end
    
    % average precision (AP)
    AP(i) = sum(P)/rel;    
end

%%
% Calculation of mean average precision (MAP)
MAP = mean(AP);

%%
% testing with text query
image_testdata = I_te';
text_testdata = T_te';

y_test = net_T(text_testdata);
classes_test = vec2ind(y_test);
text_testdata_len = length(classes_test); 

% image_output saves the corresponding image SOM node for each text test instance
image_output = zeros(1,text_testdata_len);
for i = 1:text_testdata_len
    for j = 1:length(ass_netoutput)
        if j == classes_test(i)
            image_output(i) = ass_netoutput(j);
        end
    end
end

%%
% Fetch the results for each of the text test instance
% Or we have to fetch the image input instances which have been clustered 
% in the same node as the test text instance is linked with. 

% Initialization of AP array
AP_T2I = zeros(1,text_testdata_len);

% Starting AP calculation
% It has been calculated for each test query one by one
for i = 1:text_testdata_len
    k=1;
    % Finding index numbers of those inputs which have been clustered in
    % the same cluster as the test_outcome
    for j = 1:length(classes_I)
        if image_output(i) == classes_I(j)
            indexno(k) = j;
            k = k+1;
        end
    end
    
    % initialization of similar_instances and similarity.
    % similar_instances - saves all the instances or inputs which are
    % present at index numbers mentioned in indexno array.
    % similarity array saves the euclidean distance between image instance
    % and all the retrieved results.
    similar_instances = zeros(input_I_nrows,length(indexno));
    similarity = zeros(1,length(indexno));
    for j = 1:length(indexno)
        similar_instances(:,j) = input_I(:,indexno(j));
        similarity(j) = norm(image_testdata(:,i) - similar_instances(:,j)); 
    end
    
    % calculation of mean euclidean distance
    mean_similarity = mean(similarity);
    
    % average precision calculation for a query or test instance
    P = zeros(1,length(similarity)); % precision for each outcome of a query. 
    rel = 0;  % relevance
    for j = 1:length(similarity)
        if similarity(j) > mean_similarity   % means it is relevant
            rel = rel+1;
            P(j) = rel/j;
        end
    end

    % average precision (AP)
    AP_T2I(i) = sum(P)/rel;    
end

%%
% Calculation of mean average precision (MAP)
MAP_T2I = mean(AP_T2I);

%time = toc;
