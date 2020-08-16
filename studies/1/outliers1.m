%%%this function is to exclude the outliers at all (By Marwa Ismail--April 2017)%%%%
function res1 = outliers1(R)
x= R;%[1 1 2 3 4 5 6 7 4 3 2 1 70 ];
%figure,plot(x)
x=sort(x);


%%get lower quartile
len=length(x);
l_half= x(1:round(len/2));
len1=length(l_half);
%%get the avaerage of the 2 middle points of the lower quartile
if(rem(len1,2)==0)
 Q1 =    (l_half(len1/2) + l_half((len1/2)+1))/2;
else
    
    Q1 =    (l_half(round(len1/2)));
end




u_half= x(round(len/2)+1:end);
len2=length(u_half);
%%get the avaerage of the 2 middle points of the lower quartile
if(rem(len2,2)==0)
 Q3 =    (u_half(len2/2) + u_half((len2/2)+1))/2;
else
    
    Q3 =    (u_half(round(len2/2)));
end



Quart_range = Q3-Q1;
xx=Quart_range*1.5;
y=Q1-xx;
z=Q3+xx;

xx1=Quart_range*3;
y1=Q1-xx1;
z1=Q3+xx1;
res=R;
ind=find(res>z1);

 

ind1=find(res<y1);

%res(ind1)=y1;
%figure,plot(res)

%%%Exclude outliers
ress=res;
ress(ind)=1000000;
ress(ind1)=1000000;
res1=ress(ress<1000000);

length(ind);
length(ind1);
end