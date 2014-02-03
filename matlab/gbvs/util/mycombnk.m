% so that stats toolbox is not used
function cmbs = mycombnk( nums , k )

N = length(nums);
T = N^k;
cmbs = zeros( T , k );

for j=T:-1:1       
    n = j;        
    for jj=1:k
        b = mod(n,N);
        n = n - b;
        n = n / N;
        cmbs( j , k-jj+1 ) = nums(b+1);        
        cmbs( j , : ) = sort( cmbs(j,:) );        
    end    
    is_used = zeros( N , 1 );
    for jj=1:k
        if (is_used( cmbs(j,jj)))
            cmbs(j,:) = -1 * ones(1,k);
            break;
        else
            is_used( cmbs(j,jj) ) = 1;
        end
    end               
end


cmbs = unique(cmbs,'rows');
cmbs = cmbs(2:end,:);