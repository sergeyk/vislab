function b = padImageOld( a , vpad , hpad )

if ( nargin == 2 )
  hpad = vpad;
end

if ( (size(a,1) > (2*vpad)) && (size(a,2) > (2*hpad)) )
  u = a(vpad:-1:1,:);
  b = a(end:-1:end-vpad+1,:);
  
  l = a(:,hpad:-1:1);
  r = a(:,end:-1:end-hpad+1);
  
  ul = a(vpad:-1:1,hpad:-1:1);
  ur = a(vpad:-1:1,end:-1:end-hpad+1);
  bl = a(end:-1:end-vpad+1,hpad:-1:1);
  br = a(end:-1:end-vpad+1,end:-1:end-hpad+1);
else
   u = repmat( a(1,:) , [ vpad 1 ] );
   b = repmat( a(end,:) , [ vpad 1 ] );
   
   l = repmat( a(:,1) , [ 1 hpad ] );
   r = repmat( a(:,end) , [ 1 hpad ] );
   
   ul = repmat( a(1,1) , [ vpad hpad ] );
   ur = repmat( a(1,end) , [ vpad hpad ] );
   bl = repmat( a(end,1) , [ vpad hpad ] );
   br = repmat( a(end,end) , [ vpad hpad ] );
end

b = [ ul u ur
      l  a r
      bl b br ];
