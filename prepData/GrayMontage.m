function  GrayMontage(v)

if  ndims(v)>3
    v = squeeze(v);
end
sz = size(v);
montage(reshape(single(v),[sz(1),sz(2),1,sz(3)]),'DisplayRange',[]);