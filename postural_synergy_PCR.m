clear;
clc;
%%
% ��֤PCR�ع飬��Matlab�ٷ���Աȣ�����������Эͬ����ʵ��
%%
% ����matlab�ٷ����ݼ�'spectra'
load spectra
% ����X���ݼ�Ϊԭʼ���ݼ�
X = NIR;
Y = octane;
[n,m] = size(X);
[n,p] = size(Y);
%%
% ���ùٷ�����ֱ�Ӽ������ɷݣ�ͶӰ������ݼ��Լ�Э������������ֵ
% ʹ��10�����ɷ�
[PCALoadings,PCAScores,PCAVar] = pca(X,'Economy',false);
betaPCR = regress(Y-mean(Y), PCAScores(:,1:10));
betaPCR = PCALoadings(:,1:10)*betaPCR;
betaPCR = [mean(Y) - mean(X)*betaPCR; betaPCR];
yfitPCR = [ones(n,1) X]*betaPCR;

%%
% �Զ���PCA���벢��ٷ����������-NIPALS����
X_centered = X;
Y_centered = Y;
% ��ԭʼ���ݼ������о��д���
for j = 1:m
    X_centered(:,j) =   X(:,j) - mean(X(:,j));
end
for j = 1:p
    Y_centered(:,j) =   Y(:,j) - mean(Y(:,j));
end
% ʹ��matlab����Э����������������������ֵ��ɵĶԽǾ���
[V,D] = eig(X_centered'*X_centered);
% ��������Ĳв����X_iteration
X_iteration = X_centered;
% ���屣�������ɷ�����score��loading���󣬼�ͶӰ��ľ�������ɷ־���(�������������)
num = 10;
T = zeros(n,num);
P = zeros(m,num);
eigenValues = zeros(num,1);
% ���õ������̶Ⱥʹ洢ǰһ��T(:,h)����ʱ����
tolerance = 0.000001;
t_temp = zeros(n,1);
% ���õ������α꣬����˵��PLS�ɷ����
h = 1;
% ------------------------------------------------------------------------------------------------------------
% �����õ�num���ɷ�
for h = 1:num
    % step(1)
    % ---------------------------------------------------------------------
    % ȡT(:,h)Ϊ����һ��X_centered�е����������˴�ֱ��ȡ��һ��
    T(:,h) = X_iteration(:,1);

    % step(2) to step(5)
    % ����ֱ�����������̶��ڵ����ɷ�
    while(1)
        P(:,h) = X_iteration'*T(:,h)/(T(:,h)'*T(:,h));
        % ��һ��P(:,h)
        P(:,h) =  P(:,h)/sqrt(P(:,h)'*P(:,h));
        t_temp = T(:,h);
        T(:,h) = X_iteration*P(:,h)/(P(:,h)'*P(:,h));

        % ��鵱ǰT(:,h)����һ��T(:,h)�Ƿ�����Ծ����Ƿ��������
        if max(abs(T(:,h)-t_temp)) <= tolerance
            % �洢��˳�����е�����ֵ
            eigenValues(h) = P(:,h)'*(X_centered'*X_centered)*P(:,h);
            break;
        else
        end
    end
    % ����в�������ݾ���
    % ---------------------------------------------------------------------
    X_iteration = X_iteration - T(:,h)*P(:,h)';
end
%%
% ������Լ�����������
r = n;
% ��ԭʼ���ݽ�ά�����ɷֿռ�(T)��ʹ��OLS��С���˻ع��ȡϵ������
B_inPca = inv(T'*T)*T'*Y_centered;
%B_inPca = regress(Y-mean(Y), T(:,1:num));
% ��ϵ����������ɷֿռ�ת����ԭʼ�ռ�
B_estimated = P*B_inPca;

% ������Լ����˴�ֱ��ʹ��ԭʼ���ݵ�ǰr��
X_validate = zeros(r,m);
% ��ԭʼ���ݼ�������ƽ����
for j = 1:m
    % ע�⣬�˴���ȥ��ƽ��ֵӦ��Ϊģ�����ݼ���ƽ��ֵ�����������ݵ�ƽ��ֵ
    X_validate(:,j) =   X(1:r,j) - mean(X(:,j));
end

Y_estimated = X_validate*B_estimated;
for i = 1:p
   % ע��˴����յ������Ҫ�������ݼ�Y�ľ�ֵ
   Y_estimated(:,i) = Y_estimated(:,i) + mean(Y(:,i)); 
end
%%
% ����ԭʼ������Ԥ������ݶԱ�
figure(1);
scatter(Y(:,1), yfitPCR(:,1),80,'+','b');
hold on;
% PLSR NIPALSԤ���������ԭʼ���ݽ��жԱ�
scatter(Y(1:r,1), Y_estimated(1:r,1),80,'x','r');
legend('the matlab algorithm', 'the self-defined algorithm');

%%




