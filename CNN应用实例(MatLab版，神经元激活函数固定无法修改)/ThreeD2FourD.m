function Data_4D = ThreeD2FourD(Data_3D)

[COL, ROW, NUM] = size(Data_3D);

Data_4D = zeros(COL, ROW, 1, NUM);

Data_4D(:, :, 1, :) = 1;
for i = 1:NUM
    Data_4D(:, :, 1, i) = Data_3D(:, :, i);
end