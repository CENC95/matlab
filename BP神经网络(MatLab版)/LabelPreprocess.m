function LabelofData_processed = LabelPreprocess(LabelofData)

sorted_target=sort(LabelofData,2);
label=zeros(1,1);                              
label(1,1)=sorted_target(1,1);
j=1;
for i = 2:size(LabelofData, 2)
    if sorted_target(1,i) ~= label(1,j)
        j=j+1;                              
        label(1,j) = sorted_target(1,i);
    end
end
number_class=j;

temp=zeros(number_class, size(LabelofData, 2));
for i = 1:size(LabelofData, 2)
    for j = 1:number_class
        if label(1,j) == LabelofData(1,i)
            break; 
        end
    end
    temp(j,i)=1;
end
LabelofData_processed=temp*2-1;
