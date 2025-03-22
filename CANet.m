lgraph = layerGraph();

tempLayers = imageInputLayer([299 299 3],"Name","imageinput");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],3,"Name","conv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([5 5],3,"Name","conv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = averagePooling2dLayer([4 4],"Name","avgpool2d","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = maxPooling2dLayer([4 4],"Name","maxpool","Padding","same");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat")
    convolution2dLayer([7 7],64,"Name","conv","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([5 5],128,"Name","conv_1","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    averagePooling2dLayer([4 4],"Name","avgpool2d_1","Padding","same")
    convolution2dLayer([1 1],1,"Name","conv_4","Padding","same")
    batchNormalizationLayer("Name","batchnorm_4")
    reluLayer("Name","relu_4")
    sigmoidLayer("Name","sigmoid")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],1,"Name","conv_5","Padding","same")
    sigmoidLayer("Name","sigmoid_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    averagePooling2dLayer([4 4],"Name","avgpool2d_2","Padding","same","Stride",[2 2])
    convolution2dLayer([1 1],1,"Name","conv_6","Padding","same")
    batchNormalizationLayer("Name","batchnorm_5")
    reluLayer("Name","relu_5")
    sigmoidLayer("Name","sigmoid_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],1,"Name","conv_7","Padding","same","Stride",[2 2])
    sigmoidLayer("Name","sigmoid_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_1")
    convolution2dLayer([3 3],64,"Name","conv_8","Padding","same")
    batchNormalizationLayer("Name","batchnorm_6")
    reluLayer("Name","relu_6")
    convolution2dLayer([3 3],128,"Name","conv_9","Padding","same")
    batchNormalizationLayer("Name","batchnorm_7")
    reluLayer("Name","relu_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    averagePooling2dLayer([4 4],"Name","avgpool2d_3","Padding","same")
    convolution2dLayer([1 1],1,"Name","conv_10","Padding","same")
    batchNormalizationLayer("Name","batchnorm_8")
    reluLayer("Name","relu_8")
    sigmoidLayer("Name","sigmoid_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],1,"Name","conv_11","Padding","same")
    sigmoidLayer("Name","sigmoid_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool")
    fullyConnectedLayer(64,"Name","fc")
    fullyConnectedLayer(128,"Name","fc_1")
    sigmoidLayer("Name","sigmoid_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([9 9],128,"Name","conv_12","DilationFactor",[8 8])
    batchNormalizationLayer("Name","batchnorm_9")
    reluLayer("Name","relu_9")
    fullyConnectedLayer(64,"Name","fc_2")
    fullyConnectedLayer(128,"Name","fc_3")
    sigmoidLayer("Name","sigmoid_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","conv_13","Padding","same")
    batchNormalizationLayer("Name","batchnorm_10")
    sigmoidLayer("Name","sigmoid_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_4");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = multiplicationLayer(2,"Name","multiplication_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat_1")
    convolution2dLayer([3 3],256,"Name","conv_14","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_11")
    reluLayer("Name","relu_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],512,"Name","conv_15","Padding","same","Stride",[2 2])
    batchNormalizationLayer("Name","batchnorm_12")
    reluLayer("Name","relu_11")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_1")
    fullyConnectedLayer(256,"Name","fc_4")
    fullyConnectedLayer(512,"Name","fc_5")
    sigmoidLayer("Name","sigmoid_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([9 9],512,"Name","conv_16","DilationFactor",[2 2])
    batchNormalizationLayer("Name","batchnorm_13")
    reluLayer("Name","relu_12")
    fullyConnectedLayer(256,"Name","fc_6")
    fullyConnectedLayer(512,"Name","fc_7")
    sigmoidLayer("Name","sigmoid_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_6");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_6")
    convolution2dLayer([3 3],128,"Name","conv_18","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15")
    reluLayer("Name","relu_14")
    convolution2dLayer([3 3],512,"Name","conv_19","Padding","same")
    batchNormalizationLayer("Name","batchnorm_16")
    reluLayer("Name","relu_15")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    globalAveragePooling2dLayer("Name","gapool_2")
    fullyConnectedLayer(256,"Name","fc_8")
    fullyConnectedLayer(512,"Name","fc_9")
    sigmoidLayer("Name","sigmoid_11")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([9 9],512,"Name","conv_17","DilationFactor",[2 2])
    batchNormalizationLayer("Name","batchnorm_14")
    reluLayer("Name","relu_13")
    fullyConnectedLayer(256,"Name","fc_10")
    fullyConnectedLayer(512,"Name","fc_11")
    sigmoidLayer("Name","sigmoid_12")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition_7");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    multiplicationLayer(2,"Name","multiplication_7")
    convolution2dLayer([3 3],128,"Name","conv_20","Padding","same")
    batchNormalizationLayer("Name","batchnorm_17")
    reluLayer("Name","relu_16")
    convolution2dLayer([3 3],64,"Name","conv_21","Padding","same")
    batchNormalizationLayer("Name","batchnorm_18")
    reluLayer("Name","relu_17")
    globalAveragePooling2dLayer("Name","gapool_3")
    fullyConnectedLayer(32,"Name","fc_12")
    dropoutLayer(0.5,"Name","dropout")
    fullyConnectedLayer(2,"Name","fc_13")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"imageinput","conv_2");
lgraph = connectLayers(lgraph,"imageinput","conv_3");
lgraph = connectLayers(lgraph,"imageinput","avgpool2d");
lgraph = connectLayers(lgraph,"imageinput","maxpool");
lgraph = connectLayers(lgraph,"relu","addition/in2");
lgraph = connectLayers(lgraph,"relu_1","addition/in1");
lgraph = connectLayers(lgraph,"addition","depthcat/in1");
lgraph = connectLayers(lgraph,"avgpool2d","addition_1/in2");
lgraph = connectLayers(lgraph,"maxpool","addition_1/in1");
lgraph = connectLayers(lgraph,"addition_1","depthcat/in2");
lgraph = connectLayers(lgraph,"relu_2","conv_1");
lgraph = connectLayers(lgraph,"relu_2","avgpool2d_2");
lgraph = connectLayers(lgraph,"relu_2","conv_7");
lgraph = connectLayers(lgraph,"relu_3","avgpool2d_1");
lgraph = connectLayers(lgraph,"relu_3","conv_5");
lgraph = connectLayers(lgraph,"relu_3","multiplication_1/in2");
lgraph = connectLayers(lgraph,"sigmoid","addition_2/in1");
lgraph = connectLayers(lgraph,"sigmoid_1","addition_2/in2");
lgraph = connectLayers(lgraph,"addition_2","multiplication_1/in1");
lgraph = connectLayers(lgraph,"sigmoid_2","addition_3/in1");
lgraph = connectLayers(lgraph,"sigmoid_3","addition_3/in2");
lgraph = connectLayers(lgraph,"addition_3","multiplication/in1");
lgraph = connectLayers(lgraph,"relu_7","multiplication/in2");
lgraph = connectLayers(lgraph,"multiplication","avgpool2d_3");
lgraph = connectLayers(lgraph,"multiplication","conv_11");
lgraph = connectLayers(lgraph,"multiplication","multiplication_2/in2");
lgraph = connectLayers(lgraph,"multiplication","multiplication_5/in2");
lgraph = connectLayers(lgraph,"sigmoid_4","addition_4/in1");
lgraph = connectLayers(lgraph,"sigmoid_5","addition_4/in2");
lgraph = connectLayers(lgraph,"addition_4","multiplication_2/in1");
lgraph = connectLayers(lgraph,"multiplication_2","gapool");
lgraph = connectLayers(lgraph,"multiplication_2","conv_12");
lgraph = connectLayers(lgraph,"multiplication_2","multiplication_3/in2");
lgraph = connectLayers(lgraph,"multiplication_2","conv_13");
lgraph = connectLayers(lgraph,"multiplication_2","multiplication_4/in2");
lgraph = connectLayers(lgraph,"sigmoid_6","addition_5/in1");
lgraph = connectLayers(lgraph,"sigmoid_7","addition_5/in2");
lgraph = connectLayers(lgraph,"addition_5","multiplication_3/in1");
lgraph = connectLayers(lgraph,"multiplication_3","multiplication_5/in1");
lgraph = connectLayers(lgraph,"sigmoid_8","multiplication_4/in1");
lgraph = connectLayers(lgraph,"multiplication_4","depthcat_1/in1");
lgraph = connectLayers(lgraph,"multiplication_5","depthcat_1/in2");
lgraph = connectLayers(lgraph,"relu_10","conv_15");
lgraph = connectLayers(lgraph,"relu_10","gapool_2");
lgraph = connectLayers(lgraph,"relu_10","conv_17");
lgraph = connectLayers(lgraph,"relu_11","gapool_1");
lgraph = connectLayers(lgraph,"relu_11","conv_16");
lgraph = connectLayers(lgraph,"relu_11","multiplication_6/in2");
lgraph = connectLayers(lgraph,"sigmoid_9","addition_6/in1");
lgraph = connectLayers(lgraph,"sigmoid_10","addition_6/in2");
lgraph = connectLayers(lgraph,"addition_6","multiplication_6/in1");
lgraph = connectLayers(lgraph,"sigmoid_11","addition_7/in1");
lgraph = connectLayers(lgraph,"sigmoid_12","addition_7/in2");
lgraph = connectLayers(lgraph,"addition_7","multiplication_7/in2");
lgraph = connectLayers(lgraph,"relu_15","multiplication_7/in1");