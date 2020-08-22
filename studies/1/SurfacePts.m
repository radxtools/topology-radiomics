function [Measures, I]=SurfacePts(region,dim)
[n2,n1,n3]=size(region);
[X,Y,Z] = meshgrid([1:n1].*dim(1),[1:n2].*dim(2),[1:n3].*dim(3));
region1=smooth3(region);

% todo: significance of 128??
I=isosurface(X,Y,Z,region1,128);
v=I.vertices;

len=length(v);
dim=0;
for n=1:len/2
   
    if((n*n)>len)
    
        dim=n;
        break;
    end
end

v1=zeros(dim*dim,3);
v1(1:len,:)=v;
v1(len+1:end,1)=0;
v1(len+1:end,2)=0;
v1(len+1:end,3)=0;


XX=reshape(v1(:,1),dim,dim);
YY=reshape(v1(:,2),dim,dim);
ZZ=reshape(v1(:,3),dim,dim);

[K,H,P1,P2] = CompCurvature(XX,YY,ZZ);

P11=reshape(P1,dim*dim,1);
P22=reshape(P2,dim*dim,1);
P11=P11(1:len);
P22=P22(1:len);
 
%%mean and Gaussian curvatures%%%
K1=reshape(K,dim*dim,1);
H1=reshape(H,dim*dim,1);
H1=H1(1:len);
K1=K1(1:len);
 
 %%%sharpness
 Shar=(P11-P22).^2;
 Shar1=outliers(Shar);
 Shar11=outliers1(Shar);
 
 
patch(I,'EdgeColor','none','FaceColor','interp','FaceVertexCData',Shar1);colorbar;


 
 SHH=zeros(4,1);
 SHH(1,1)=mean(Shar11);
 SHH(2,1)=median(Shar11);
 SHH(3,1)=kurtosis(Shar11);
 SHH(4,1)=std(Shar11);



 Measures=[SHH];
% save('Measures_region','Measures_region')


 end
