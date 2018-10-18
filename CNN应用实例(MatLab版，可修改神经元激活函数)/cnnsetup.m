function net = cnnsetup(net, x, y)
    inputmaps = 1;
    mapsize = size(squeeze(x(:, :, 1)));

    for l = 1 : numel(net.layers)   %  layer
        if strcmp(net.layers{l}.type, 's') % �������� �Ӳ�����
                      
			mapsize = floor(mapsize / net.layers{l}.scale);
            for j = 1 : inputmaps % inputmap������һ���ж���������ͼ
                net.layers{l}.b{j} = 0; % ��ƫ�ó�ʼ��Ϊ0
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
                    net.layers{l}.b{j} = 0; % ��ƫ�ó�ʼ��Ϊ0
                    
                    
                    net.layers{l}.k{i}{j} = Temp(:, :, j);
                end
            
            end
            
            inputmaps = net.layers{l}.outputmaps; 
        end
    end
	
	% fvnum ��������ǰ��һ�����Ԫ������
	% ��һ�����һ���Ǿ���pooling��Ĳ㣬������inputmaps������map��ÿ������map�Ĵ�С��mapsize��
	% ���ԣ��ò����Ԫ������ inputmaps * ��ÿ������map�Ĵ�С��
	% prod: Product of elements.
	% For vectors, prod(X) is the product of the elements of X
	% ������ mapsize = [����map������ ����map������]������prod����� ����map����*��
    fvnum = prod(mapsize) * inputmaps;
	% onum �Ǳ�ǩ�ĸ�����Ҳ�����������Ԫ�ĸ�������Ҫ�ֶ��ٸ��࣬��Ȼ���ж��ٸ������Ԫ
    onum = size(y, 1);

	% ���������һ����������趨
	% ffb �������ÿ����Ԫ��Ӧ�Ļ�biases
    net.ffb = zeros(onum, 1);
	% ffW �����ǰһ�� �� ����� ���ӵ�Ȩֵ��������֮����ȫ���ӵ�
    net.ffW = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum));
end
