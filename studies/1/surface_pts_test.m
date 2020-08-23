dim = [1 1 1]
A = zeros(5,5,5);
A(2:4,2:4,2:4) = 1;


[n2,n1,n3]=size(A);
[X,Y,Z] = meshgrid([1:n1].*dim(1),[1:n2].*dim(2),[1:n3].*dim(3));
R=smooth3(A);

% todo: significance of 128??
I=isosurface(R);
v=I.vertices;

v = zeros(10,10)
len=length(v);
dim=0;
for n=1:len/2
   
    if((n*n)>len)
    
        dim=n;
        break;
    end
end

v1=zeros(dim*dim,3);
%v1(1:len,:)=v;
%v1(len+1:end,1)=0;
%v1(len+1:end,2)=0;
%v1(len+1:end,3)=0;