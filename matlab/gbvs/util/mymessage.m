function mymessage(param,s,varargin)
if (param.verbose == 1 )    
    s = [ '\t' s ];
    if ( strcmp(param.verboseout,'screen') == 1 )        
        if ( nargin == 2 )
            fprintf(s);
        else
            fprintf(s,varargin{:});
        end
    else
        fido = fopen(param.verboseout,'a');
        if ( nargin == 2 )
            fprintf(fido,s);
        else
            fprintf(fido,s,varargin{:});
        end
        fclose(fido);
    end
end        