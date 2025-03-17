files = {
'Newman/karate'
'ND/nd3k'
'ND/nd6k'
'ND/nd12k'
'ND/nd24k'
'Freescale/Freescale2'
'LAW/indochina-2004'
'SNAP/com-Amazon'
'SNAP/com-Orkut'
'SNAP/com-Youtube'
'SNAP/com-LiveJournal'
'SNAP/roadNet-CA'
'SNAP/roadNet-PA'
'SNAP/roadNet-TX'
'SNAP/soc-Pokec'
'GAP/GAP-road' } ;

files = {
'Freescale/Freescale2' } ;

files = {
'LAW/indochina-2004'
'SNAP/com-Amazon'
'SNAP/com-Orkut'
'SNAP/com-Youtube'
'SNAP/com-LiveJournal'
'SNAP/roadNet-CA'
'SNAP/roadNet-PA'
'SNAP/roadNet-TX'
'SNAP/soc-Pokec'
'GAP/GAP-road'
'GAP/GAP-twitter'
'GAP/GAP-web'
}

%{
'GAP/GAP-kron'
'GAP/GAP-urand' } ;
%}

for k = 1:length (files)

    % load the *.mat file
    file = files {k} ;
    Prob = ssget (file)
    A = Prob.A ;
    [m n] = size (A) ;
    if (isfield (Prob, 'Zeros'))
        Z = Prob.Zeros ;
    else
        Z = [ ] ;
    end
    clear Prob

    % load the *.triplet file
    i = find (file == '/') ;
    tfile = ['../MM/' file '/' file(i+1:end) '.triplet'] ;
    T = load (tfile) ;
    S = sparse (T(:,1)+1, T(:,2)+1, T (:,3), m, n) ;

    % make sure they match
    assert (isequal (S, A)) ;
    clear S

    if (~isempty (Z))
        z = find (T (:,3) == 0) ;
        X = ones (length (z), 1) ;
        S = sparse (T (z,1)+1, T (z,2)+1, X, m, n) ;
        assert (isequal (S, Z)) ;
        clear X z
    end

    clear S A Z T

end

