function [output,cost] = mNMF(input, num_bases, fft_win, shift_len, max_iter)

% Multichannel NMF implementation for stereo signals

% input: mixed audio signal to be separated (stereo)
% num_bases: number of bases to use in NMF
% fft_win: window length for STFT
% shift_len: shift length for STFT 
% max_iter: number of iterations to run algorithm 

% output: separated audio signals (3D matrix)
% cost: vector of cost function values at each iteration


% separating 2 audio sources 
num_sources = 2;


% Pre-processing using short time Fourier transform (save stft frequencies
% for later ISTFT)
[X, stft_freq] = STFT(input,fft_win,shift_len,'hamming');

% normalize signal (save sum for later ISTFT)
sig_norm = sum(mean(mean(abs(X).^2,3),2),1);
X = X./sig_norm;
[I,J,M] = size(X);

x = permute(X,[3,1,2]);

XX = zeros(I,J,M,M);
for i = 1:I
    for j = 1:J
        XX(i,j,:,:) = x(:,i,j)*x(:,i,j)' + eye(M)*(10^(-12)); 
    end
end

% multichannel NMF algorithm 

% set random initial matrices T, V, H
T = max(rand(I,num_bases),eps);
V = max(rand(num_bases,J),eps);
H = repmat(sqrt(eye(M)/M),[1,1,I,num_sources]);
H = permute(H,[3,4,1,2]);
Z = max( ((0.01)*rand(num_bases,num_sources) + 1/num_sources)./sum(((0.01)*rand(num_bases,num_sources) + 1/num_sources),2), eps );

% Initialize Xhat
Xhat = zeros(I,J,M,M);
for m = 1:M*M
    H_temp = H(:,:,m); % I x N
    Xhat(:,:,m) = ((H_temp*Z').*T)*V;
end

% Initialize cost vector for output
cost = zeros( max_iter+1, 1 );
cost(1) = cost_fun( XX, Xhat, I, J, M );

% main loop for max_iter iterations
for iter = 1:max_iter
    [ Xhat, T, V, H, Z ] = update( XX, Xhat, T, V, H, Z, I, J, num_bases, num_sources, M );
    cost(iter+1) = cost_fun( XX, Xhat, I, J, M );
end


% wiener filter
Y = zeros(I,J,M,num_sources);
Xhat = permute(Xhat,[3,4,1,2]); 
for i = 1:I
    for j = 1:J
        for source_idx = 1:num_sources
            y_sig = 0;
            for k = 1:num_bases
                y_sig = y_sig + Z(k,source_idx)*T(i,k)*V(k,j);
            end
            Y(i,j,:,source_idx) = y_sig * squeeze(H(i,source_idx,:,:))/Xhat(:,:,i,j)*x(:,i,j); 
        end
    end
end

% Inverse short time Fourier transform output
Y = Y.*sig_norm; 
for source_idx = 1:num_sources
    output(:,:,source_idx) = ISTFT(Y(:,:,:,source_idx), shift_len, stft_freq, size(input,1));
end

end



function [J] = cost_fun( X, Xhat, I, J, M )
Xhat_inv = inv_2( Xhat, I, J, M );
XXhat_inv = mult_2( X, Xhat_inv, I, J, M );
trXXhat_inv = trace_2( XXhat_inv, I, J, M );
detXXhat_inv = det_2( XXhat_inv, I, J, M );
J = real(trXXhat_inv) - log(real(detXXhat_inv)) - M;
J = sum(J(:));
end

function [ Xhat, T, V, H, Z ] = update( X, Xhat, T, V, H, Z, I, J, K, N, M )
% perform update for T matrix 
Xhat_inv = inv_2( Xhat, I, J, M );
Xhat_invX = mult_2( Xhat_inv, X, I, J, M );
Xhat_invXXhat_inv = mult_2( Xhat_invX, Xhat_inv, I, J, M );
T_num = T_fun( Xhat_invXXhat_inv, V, Z, H, I, K, M );
T_denom = T_fun( Xhat_inv, V, Z, H, I, K, M ); 
T = T.*max(sqrt(T_num./T_denom),eps);
Xhat = zeros(I,J,M,M);
for m = 1:M*M
    H_temp = H(:,:,m); 
    Xhat(:,:,m) = ((H_temp*Z').*T)*V;
end

% perform udpate for V matrix
Xhat_inv = inv_2( Xhat, I, J, M );
Xhat_invX = mult_2( Xhat_inv, X, I, J, M );
Xhat_invXXhat_inv = mult_2( Xhat_invX, Xhat_inv, I, J, M );
V_num = V_fun( Xhat_invXXhat_inv, T, Z, H, J, K, M ); 
V_denom = V_fun( Xhat_inv, T, Z, H, J, K, M ); 
V = V.*max(sqrt(V_num./V_denom),eps);
Xhat = zeros(I,J,M,M);
for m = 1:M*M
    H_temp = H(:,:,m); 
    Xhat(:,:,m) = ((H_temp*Z').*T)*V;
end

% perform update for Z matrix 
Xhat_inv = inv_2( Xhat, I, J, M );
Xhat_invX = mult_2( Xhat_inv, X, I, J, M );
Xhat_invXXhat_inv = mult_2( Xhat_invX, Xhat_inv, I, J, M );
Z_num = Z_fun( Xhat_invXXhat_inv, T, V, H, K, N, M ); 
Z_denom = Z_fun( Xhat_inv, T, V, H, K, N, M ); 
Z = Z.*sqrt(Z_num./Z_denom);
Z = max( Z./sum(Z,2), eps ); 
Xhat = zeros(I,J,M,M);
for m = 1:M*M
    H_temp = H(:,:,m); 
    Xhat(:,:,m) = ((H_temp*Z').*T)*V;
end

% perform update for H matrix (using riccati equation solver)
Xhat_inv = inv_2( Xhat, I, J, M );
Xhat_invX = mult_2( Xhat_inv, X, I, J, M );
Xhat_invXXhat_inv = mult_2( Xhat_invX, Xhat_inv, I, J, M );
H = riccati( Xhat_invXXhat_inv, Xhat_inv, T, V, H, Z, I, J, N, M );
Xhat = zeros(I,J,M,M);
for m = 1:M*M
    H_temp = H(:,:,m); 
    Xhat(:,:,m) = ((H_temp*Z').*T)*V;
end


end

% update function helpers 
function [T_new] = T_fun( X, V, Z, H, I, K, M )
T_new = zeros(I,K);
for m = 1:M*M
    T_new = T_new + real( (X(:,:,m)*V').*(conj(H(:,:,m))*Z') ); 
end
end

function [V_new] = V_fun( X, T, Z, H, J, K, M )
V_new = zeros(K,J);
for m = 1:M*M
    V_new= V_new + real( ((H(:,:,m)*Z').*T)'*X(:,:,m) ); 
end
end

function [Z_new] = Z_fun( X, T, V, H, K, N, M)
Z_new = zeros(K,N);
for m = 1:M*M
    Z_new = Z_new + real( ((X(:,:,m)*V.').*T)'*H(:,:,m) ); 
end
end

% Riccati equation
function [H] = riccati(X, Y, T, V, H, Z, I, J, N, M)
X = reshape(permute(X, [3 4 2 1]), [M*M, J, I]); 
Y = reshape(permute(Y, [3 4 2 1]), [M*M, J, I]);
for n = 1:N 
    for i = 1:I
        ZTV = (T(i,:).*Z(:,n)')*V;
        A = reshape(Y(:,:,i)*ZTV', [M, M]);
        B = reshape(X(:,:,i)*ZTV', [M, M]);
        Hin = reshape(H(i,n,:,:), [M, M]);
        C = Hin*B*Hin;
        AC = [zeros(M), -1*A; -1*C, zeros(M)];
        [eig_vector, eig_value] = eig(AC);
        idx = find(diag(eig_value)<0);
        F = eig_vector(1:M,idx);
        G = eig_vector(M+1:end,idx);
        Hin = G/F;
        Hin = (Hin+Hin')/2 + eye(M)*(10^(-12));
        H(i,n,:,:) = Hin/trace(Hin);
    end
end
end

%%%%%%%%
% Fast matrix computation functions for 2 dimensions (stereo audio)
% (by recommendation of paper by H. Sawada et. al)

% fast matrix inverse 
function [inv] = inv_2( X, I, J, M )
inv = zeros(I,J,M,M);
det = X(:,:,1,1).*X(:,:,2,2) - X(:,:,1,2).*X(:,:,2,1);
inv(:,:,1,1) = X(:,:,2,2);
inv(:,:,1,2) = -1*X(:,:,1,2);
inv(:,:,2,1) = conj(inv(:,:,1,2));
inv(:,:,2,2) = X(:,:,1,1);
inv = inv./det; 
end

% fast matrix multiply
function [XY] = mult_2( X, Y, I, J, M )
XY = zeros( I, J, M, M );
XY(:,:,1,1) = X(:,:,1,1).*Y(:,:,1,1) + X(:,:,1,2).*Y(:,:,2,1);
XY(:,:,1,2) = X(:,:,1,1).*Y(:,:,1,2) + X(:,:,1,2).*Y(:,:,2,2);
XY(:,:,2,1) = X(:,:,2,1).*Y(:,:,1,1) + X(:,:,2,2).*Y(:,:,2,1);
XY(:,:,2,2) = X(:,:,2,1).*Y(:,:,1,2) + X(:,:,2,2).*Y(:,:,2,2);
end

% fast trace
function [tr] = trace_2( X, I, J, M )
tr = X(:,:,1,1) + X(:,:,2,2);
end

% fast determinant 
function [det] = det_2( X, I, J, M )
det = X(:,:,1,1).*X(:,:,2,2) - X(:,:,1,2).*X(:,:,2,1);
end






