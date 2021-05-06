clear;
clc;
%%
% ��֤PLSR�ع飬��Matlab�ٷ���Աȣ�����������Эͬ����ʵ��
%%
% ����matlab�ٷ������������ݣ���������ʵ��
load spectra
%%
% �������ݱ�����ȷ����Ҫ��PLS�ɷָ���
X = NIR;
Y = octane;
num = 10;
[n,m] = size(X);
[n,p] = size(Y);
% ʹ��matlab����PLSR����PLSR�ع�
[Xloadings,Yloadings,Xscores,Yscores,betaPLS,PLSPctVar] = plsregress(X,Y,num);
% ����PLS�ɷݶԻع���Ч�԰ٷֱȵ�Ӱ��
%figure(1);
%plot(1:10,cumsum(100*PLSPctVar(2,:)),'-bo');
% xlabel('Number of PLS components');
% ylabel('Percent Variance Explained in Y');
%%
% �Զ���PLSR��������ݽ��лع��������ǰһ�����ݶԱ�
% �˴�ʹ��NIPALS�㷨,
% Matlab�ڲ�ʹ��SIMPLS�㷨�������ڶԵ�����Y���лع�ʱû�б������𣬹ʿ����ø�������֤�������ڶ������Y���ݣ����ݴ��ڲ���

% ��ԭʼ��������ƽ�����У���������һ��(�˴������һ��)
X_centered = zeros(n,m);
Y_centered = zeros(n,p);
for j = 1:m
    X_centered(:,j) =   X(:,j) - mean(X(:,j));
end
for j = 1:p
    Y_centered(:,j) =   Y(:,j) - mean(Y(:,j));
end

% ��ʼ��������Score��Loading

% ���õ������̶Ⱥʹ洢ǰһ��T(:,h)����ʱ����
tolerance = 0.001;
t_temp = zeros(n,1);
% ���õ������α꣬����˵��PLS�ɷ����
h = 1;
% ��ʼ��score��loading�����С
T = zeros(n,num); % X score
P = zeros(m,num); % X loading
W = P;            % ���ڴ���P��ʹT��������������
U = zeros(n,num); % Y score
Q = zeros(p,num); % Y loading
B = zeros(n,1);   % �ع�ϵ������

% �ع��Եõ������ع���� Model buildig
% ------------------------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------------------------
% �����õ�num���ɷ�
for h = 1:num
    % step(1)
    % ---------------------------------------------------------------------
    % ȡu_hΪ����һ��Y_centered�е����������˴�ֱ��ȡ��һ��
    U(:,h) = Y_centered(:,1);
    
    % step(2) to step(8)
    % ---------------------------------------------------------------------
    while 1
        % �����ݾ���X_centered��
        W(:,h) = X_centered'*U(:,h)/(U(:,h)'*U(:,h));
        % �����ݽ��й�һ��
        W(:,h) = W(:,h)/sqrt(W(:,h)'*W(:,h));
        t_temp = T(:,h);
        T(:,h) = X_centered*W(:,h)/(W(:,h)'*W(:,h));

        % �����ݾ���Y_centered��
        Q(:,h) = Y_centered'*T(:,h)/(T(:,h)'*T(:,h));
        % �����ݽ��й�һ��
        Q(:,h) = Q(:,h)/sqrt(Q(:,h)'*Q(:,h));
        U(:,h) = Y_centered*Q(:,h)/(Q(:,h)'*Q(:,h));

        % ���T(:,h)��T(:,h)��ǰһ���Ƿ���ȣ���С��ĳ����ֵ���PLS�ɷݵ�����ɣ����򷵻ؼ�������
        if max(abs(T(:,h)-t_temp)) <= tolerance
            break;
        else
        end
    end
    
    % step(9) to step(13)
    % ---------------------------------------------------------------------
    P(:,h) = X_centered'*T(:,h)/(T(:,h)'*T(:,h));
    % �����ݽ��й�һ��
    p_norm = sqrt(P(:,h)'*P(:,h));
    P(:,h) = P(:,h)/p_norm;
    T(:,h) = T(:,h)*p_norm;
    W(:,h) = W(:,h)*p_norm;
    B(h) = U(:,h)'*T(:,h)/(T(:,h)'*T(:,h));
    
    % ����в�������ݾ���
    % ---------------------------------------------------------------------
    X_centered = X_centered - T(:,h)*P(:,h)';
    Y_centered = Y_centered - B(h)*T(:,h)*Q(:,h)';
end

% Ԥ�� Prediction
% ------------------------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------------------------
% ������һ����֤���ݼ�X_validate,��������Ϊr��������Ϊ��Ϊm�������Ϲ�����ģ�ͼ���Ԥ���Y
% �˴�ʹ��ԭʼ���ݣ���
num = 10;
r = n;
T_est = zeros(r,num);
X_validate = zeros(r,m);
Y_estimated = zeros(r,p);
% ��ԭʼ���ݼ�������ƽ����
for j = 1:m
    % ע�⣬�˴���ȥ��ƽ��ֵӦ��Ϊģ�����ݼ���ƽ��ֵ�����������ݵ�ƽ��ֵ
    X_validate(1:r,j) =   X(1:r,j) - mean(X(:,j));
end

% ����Ԥ���T
for h = 1:num
    T_est(:,h) = X_validate*W(:,h);
    X_validate = X_validate - T_est(:,h)*P(:,h)';
end

% ����Ԥ���Y
for h = 1:num
    Y_estimated = Y_estimated + B(h)*T_est(:,h)*Q(:,h)';
end
for i = 1:p
   % ע��˴����յ������Ҫ�������ݼ�Y�ľ�ֵ
   Y_estimated(:,i) = Y_estimated(:,i) + mean(Y(:,i)); 
end

% ------------------------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------------------------
% ����SIMPLSԤ���Yֵ��ԭʼYֵ�Ա�
figure(2);
yfitPLS = [ones(n,1) X]*betaPLS;
scatter(Y(:,1), yfitPLS(:,1),80,'+','b');
hold on;
% PLSR NIPALSԤ���������ԭʼ���ݽ��жԱ�
scatter(Y(1:r,1), Y_estimated(1:r,1),80,'x','r');
legend('the SIMPLS algorithm', 'the NIPALS algorithm');

xlabel('the validate Y data');
ylabel('the estimated Y data');
%%

