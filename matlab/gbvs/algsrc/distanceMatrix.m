function dx = distanceMatrix( dims , lx , cyclic_type )

%% some useful numbers to access nodes at each resolution level
d = prod(dims,2);
N = sum(d);
dx = zeros(N,N);
Nmaps = size(dims,1);
offsets = zeros(Nmaps,1);
for i=2:Nmaps
    offsets(i) = d(i-1) + offsets(i-1);
end
    
sd = {};
for i=1:Nmaps
    sd{i} = simpledistance( dims(i,:) , cyclic_type );
end

%% form clique on all nodes in same scale
for i=1:Nmaps    
    mapsize = d(i);
    dmatrix = sd{i};
    for ii=1:mapsize
        for jj=1:mapsize
            dx( offsets(i)+ii, offsets(i)+jj ) = dmatrix(ii,jj) * d(1)/mapsize;
        end
    end    
end

map_pairs = mycombnk( [ 1 : Nmaps ] , 2);
dmatrix = sd{1};
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

            nla = length(la);
            nlb = length(lb);
            
            %% convention betweeen 0-index and 1-index
            la = la + 1; 
            lb = lb + 1;
            
            mean_dist = 0;
	    max_dist = -inf;
            for iii=1:nla
                for jjj=1:nlb
		    dd = dmatrix( la(iii) , lb(jjj) );
                    mean_dist = mean_dist + dd;
		    max_dist = max( dd , max_dist );
                end
            end
            
            mean_dist = mean_dist / (nla*nlb);            
            dx( nodes_a(ii) , nodes_b(jj) ) = mean_dist; %max_dist; %mean_dist;
            dx( nodes_b(jj) , nodes_a(ii) ) = mean_dist; %max_dist; %mean_dist;
            
        end % end jj=1:nb
    end % end ii=1:na
end % end i=1:size(map_pairs,1)
