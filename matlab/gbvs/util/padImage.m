function b = padImage( a , vpad , hpad )

if ( nargin == 2 )
  hpad = vpad;
end

u = repmat( a(1,:) , [ vpad 1 ] );
b = repmat( a(end,:) , [ vpad 1 ] );

l = repmat( a(:,1) , [ 1 hpad ] );
r = repmat( a(:,end) , [ 1 hpad ] );

ul = repmat( a(1,1) , [ vpad hpad ] );
ur = repmat( a(1,end) , [ vpad hpad ] );
bl = repmat( a(end,1) , [ vpad hpad ] );
br = repmat( a(end,end) , [ vpad hpad ] );

b = [ ul u ur
      l  a r
      bl b br ];
