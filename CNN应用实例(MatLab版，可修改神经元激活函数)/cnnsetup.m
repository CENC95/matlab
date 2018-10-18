function net = cnnsetup(net, x, y)
    inputmaps = 1;
    mapsize = size(squeeze(x(:, :, 1)));

    for l = 1 : numel(net.layers)   %  layer
        if strcmp(net.layers{l}.type, 's') % 如果这层是 子采样层
                      
			mapsize = floor(mapsize / net.layers{l}.scale);
            for j = 1 : inputmaps % inputmap就是上一层有多少张特征图
                net.layers{l}.b{j} = 0; % 将偏置初始化为0
            end
        end
        if strcmp(net.layers{l}.type, 'c') 
            
			mapsize = mapsize - net.layers{l}.kernelsize + 1;
            
            fan_out = net.layers{l}.outputmaps * net.layers{l}.kernelsize ^ 2;
            for i = 1:inputmaps
                Temp = randn([net.layers{l}.kernelsize, net.layers{l}.kernelsize, net.layers{l}.outputmaps]) .* 0.01 + 0;
                for j = 1 : net.layers{l}.outputmaps

                    fan_in = inputmaps * net.layers{l}.kernelsize ^ 2;
                    %{
                    for i = 1 : inputmaps                     
                        net.layers{l}.k{i}{j} = (rand(net.layers{l}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                    end
                    %}
                    net.layers{l}.b{j} = 0; % 将偏置初始化为0
                    
                    
                    net.layers{l}.k{i}{j} = Temp(:, :, j);
                end
            
            end
            
            inputmaps = net.layers{l}.outputmaps; 
        end
    end
	
	% fvnum 是输出层的前面一层的神经元个数。
	% 这一层的上一层是经过pooling后的层，包含有inputmaps个特征map。每个特征map的大小是mapsize。
	% 所以，该层的神经元个数是 inputmaps * （每个特征map的大小）
	% prod: Product of elements.
	% For vectors, prod(X) is the product of the elements of X
	% 在这里 mapsize = [特征map的行数 特征map的列数]，所以prod后就是 特征map的行*列
    fvnum = prod(mapsize) * inputmaps;
	% onum 是标签的个数，也就是输出层神经元的个数。你要分多少个类，自然就有多少个输出神经元
    onum = size(y, 1);

	% 这里是最后一层神经网络的设定
	% ffb 是输出层每个神经元对应的基biases
    net.ffb = zeros(onum, 1);
	% ffW 输出层前一层 与 输出层 连接的权值，这两层之间是全连接的
    net.ffW = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum));
end
