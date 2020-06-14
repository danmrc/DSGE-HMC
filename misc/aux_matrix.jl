using LinearAlgebra

function duplication_matrix(n)
    mm = ones(n,n)
    tt = trues(n,n)
    uptt = UpperTriangular(tt)
    lott = LowerTriangular(tt)
    dd = Diagonal(tt)
    uptt = uptt .⊻ dd
    ind = 1:(n+1)*n/2
    mm[lott] = ind

    lott = lott .⊻ dd

    mm[uptt] = mm[lott]'
    mm = reshape(mm,n^2,1)
    m_fim = [mm[i] == ind[j] for i = 1:size(mm,1),j = 1:size(ind,1)]
    return m_fim
end


function vech(A)
    if !issymmetric(A)
        error("Not Symmetric")
    end
    n = size(A,1)
    tt = trues(n,n)
    n = Int(n*(n+1)/2)
    lott = LowerTriangular(tt)
    A_vec = A[lott]
    return reshape(A_vec,n,1)
end

function vecc(A)
    n = size(A,1)*size(A,2)
    return reshape(A',n,1)
end

function commutation_matrix(m,n)
    mat = zeros(m*n,m*n)
    seq = 1:(m*n)
    ind = mod.(seq .-1,m)*n + floor.((seq .-1)/m) .+1
    ind = Int.(ind)
    for k = 1:(m*n)
        mat[k,ind[k]] = 1
    end
    return mat
end
