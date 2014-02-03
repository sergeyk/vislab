function pnew = getBestRows( p )

% given collection of p = [ a b ; a1 b1 ; a2 b2 .. ]
% trims out instances where bi = bj , choosing row with maximum a.

bs = unique(p(:,2));
Nbs = length(bs);
pnew = zeros(Nbs,2);
for i=1:Nbs,
    pnew(i,:) = [ max(p(p(:,2)==bs(i),1)) bs(i) ];
end