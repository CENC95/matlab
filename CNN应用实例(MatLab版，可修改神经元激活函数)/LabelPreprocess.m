function LabelofData_processed = LabelPreprocess(LabelofData, NumofLabel)

temp=zeros(NumofLabel, size(LabelofData, 2));
for i = 1:size(LabelofData, 2)
    temp(LabelofData(i),i)=1;
end
LabelofData_processed=temp;%temp*2-1;
