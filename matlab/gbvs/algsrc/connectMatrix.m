function cx = connectMatrix( dims , lx , inter_type , intra_type , cyclic_type )

% inter_type :
%   1 => only neighbor
%   2 => everywhere inter-scale on consecutive scales
% intra_type :
%   1 => only neighbor
%   2 => connect to everynode on same scale
% cyclic_type
%   1 => cyclic boundary rules
%   2 => non-cyclic boundaries

%% some useful numbers to access nodes at each resolution level
d = prod(dims,2);
N = sum(d);
cx = zeros(N,N);
Nmaps = size(dims,1);
offsets = zeros(Nmaps,1);
for i=2:Nmaps
    offsets(i) = d(i-1) + offsets(i-1);
end
    
%% connect nodes on a single resolution/level
for i=1:Nmaps    
    mapsize = d(i);
    if ( intra_type == 1 ) %% only neighbors and self
        dmatrix = simpledistance( dims(i,:) , cyclic_type );
        dmatrix = (dmatrix <= 1);
        cx( (offsets(i)+1):(offsets(i)+mapsize), (offsets(i)+1):(offsets(i)+mapsize) ) = dmatrix;
    else %% connect everyone on 
        cx( (offsets(i)+1):(offsets(i)+mapsize), (offsets(i)+1):(offsets(i)+mapsize) ) = 1;
    end
end

%%%
%% for inter-scale nodes , connect according to some rule
%% inter_type is 1  ==> connect only nodes corresponding to same location
%% inter_type is 2  ==> connect nodes corresponding to different locations
%%%

map_pairs = mycombnk( [ 1 : Nmaps ] , 2);
for i=1:size(map_pairs,1)
    
    mapi1 = map_pairs(i,1);
    mapi2 = map_pairs(i,2);
    
    %% nodes in map at level mapi1
    nodes_a = [ offsets(mapi1) + 1 : offsets(mapi1) + d(mapi1) ];

    %% nodes in map at level mapi2
    nodes_b = [ offsets(mapi2) + 1 : offsets(mapi2) + d(mapi2) ];

    %% for each pair , possibly connect
    for ii=1:d(mapi1)
        for jj=1:d(mapi2)
            %% using location matrix, determine locations of the 
            %% two nodes            
            la = lx( nodes_a(ii) , 3:(2+lx(nodes_a(ii),2)) );
            lb = lx( nodes_b(jj) , 3:(2+lx(nodes_b(jj),2)) );
            
            if ( inter_type == 1 ) %% only connect inter-scale nodes
                                   %% which correspond to same location
                if ( length( intersect( la , lb ) ) > 0 )
                    cx( nodes_a(ii) , nodes_b(jj) ) = 1;
                    cx( nodes_b(jj) , nodes_a(ii) ) = 1;
                end
            elseif ( inter_type == 2) %% connect all inter-scale nodes
                cx( nodes_a(ii) , nodes_b(jj) ) = 1;
                cx( nodes_b(jj) , nodes_a(ii) ) = 1;                                                               
            end
        end % end jj=1:nb
    end % end ii=1:na
end % end i=1:size(map_pairs,1)

cx(cx==0) = inf;
