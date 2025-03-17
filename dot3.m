
% only works if the matrix is symmetric
files = {
'Newman/karate'
'GAP/GAP-kron'
'GAP/GAP-urand' } ;

for k = 1:length (files)

    % load the *.mat file
    file = files {k} ;
    Prob = ssget (file)
    A = Prob.A ;
    [m n] = size (A) ;
    if (isfield (Prob, 'Zeros'))
        error ('!') ;
    end
    clear Prob

    % create the *.triplet file
    i = find (file == '/') ;
    tfile = ['../MM/' file '/' file(i+1:end) '.triplet']

    A = logical (A) ;
    fprintf ('A now logical\n') ;

    [I,J,X] = find (A) ;
    clear A
    I = I-1 ;
    J = J-1 ;
    T = [J I X]' ;
    fprintf ('got T\n') ;
    clear I J X
    f = fopen (tfile, 'w') ;
    fprintf (f, '%d %d %g\n', T) ;
    fclose (f) ;
    clear T 

end

