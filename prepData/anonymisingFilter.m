function  v = anonymisingFilter(v, scale, blur, bin)

if nargin<4, bin=false; end

if blur
    v = imgaussfilt3(medfilt3(v),5);
end

v = mat2gray(imresize3(v,scale,'cubic'));

v = uint8(v*255);

if bin
    v = uint8(v>=0.5);
end


