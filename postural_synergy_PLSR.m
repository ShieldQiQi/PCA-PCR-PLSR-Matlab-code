clear;
clc;
%%
% 验证PLSR回归，与Matlab官方库对比，并用于手势协同分析实验
%%
% 加载matlab官方汽油辛烷数据，用于样例实验
load spectra
%%
% 建立数据变量，确定需要的PLS成分个数
X = NIR;
Y = octane;
num = 10;
[n,m] = size(X);
[n,p] = size(Y);
% 使用matlab内置PLSR进行PLSR回归
[Xloadings,Yloadings,Xscores,Yscores,betaPLS,PLSPctVar] = plsregress(X,Y,num);
% 分析PLS成份对回归有效性百分比的影响
%figure(1);
%plot(1:10,cumsum(100*PLSPctVar(2,:)),'-bo');
% xlabel('Number of PLS components');
% ylabel('Percent Variance Explained in Y');
%%
% 自定义PLSR代码对数据进行回归分析，与前一步数据对比
% 此处使用NIPALS算法,
% Matlab内部使用SIMPLS算法，两者在对单变量Y进行回归时没有本质区别，故可以用该数据验证，但对于多变量的Y数据，数据存在差异

% 对原始数据样本平均居中，并比例归一化(此处无需归一化)
X_centered = zeros(n,m);
Y_centered = zeros(n,p);
for j = 1:m
    X_centered(:,j) =   X(:,j) - mean(X(:,j));
end
for j = 1:p
    Y_centered(:,j) =   Y(:,j) - mean(Y(:,j));
end

% 开始迭代计算Score和Loading

% 设置迭代容忍度和存储前一步T(:,h)的临时变量
tolerance = 0.001;
t_temp = zeros(n,1);
% 设置迭代的游标，或者说是PLS成份序号
h = 1;
% 初始化score和loading矩阵大小
T = zeros(n,num); % X score
P = zeros(m,num); % X loading
W = P;            % 用于代替P，使T保持正交的属性
U = zeros(n,num); % Y score
Q = zeros(p,num); % Y loading
B = zeros(n,1);   % 回归系数向量

% 回归以得到各个回归参数 Model buildig
% ------------------------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------------------------
% 迭代得到num个成份
for h = 1:num
    % step(1)
    % ---------------------------------------------------------------------
    % 取u_h为任意一个Y_centered中的列向量，此处直接取第一列
    U(:,h) = Y_centered(:,1);
    
    % step(2) to step(8)
    % ---------------------------------------------------------------------
    while 1
        % 在数据矩阵X_centered中
        W(:,h) = X_centered'*U(:,h)/(U(:,h)'*U(:,h));
        % 对数据进行归一化
        W(:,h) = W(:,h)/sqrt(W(:,h)'*W(:,h));
        t_temp = T(:,h);
        T(:,h) = X_centered*W(:,h)/(W(:,h)'*W(:,h));

        % 在数据矩阵Y_centered中
        Q(:,h) = Y_centered'*T(:,h)/(T(:,h)'*T(:,h));
        % 对数据进行归一化
        Q(:,h) = Q(:,h)/sqrt(Q(:,h)'*Q(:,h));
        U(:,h) = Y_centered*Q(:,h)/(Q(:,h)'*Q(:,h));

        % 检查T(:,h)与T(:,h)的前一步是否相等，若小于某个数值则该PLS成份迭代完成，否则返回继续迭代
        if max(abs(T(:,h)-t_temp)) <= tolerance
            break;
        else
        end
    end
    
    % step(9) to step(13)
    % ---------------------------------------------------------------------
    P(:,h) = X_centered'*T(:,h)/(T(:,h)'*T(:,h));
    % 对数据进行归一化
    p_norm = sqrt(P(:,h)'*P(:,h));
    P(:,h) = P(:,h)/p_norm;
    T(:,h) = T(:,h)*p_norm;
    W(:,h) = W(:,h)*p_norm;
    B(h) = U(:,h)'*T(:,h)/(T(:,h)'*T(:,h));
    
    % 计算残差，更新数据矩阵
    % ---------------------------------------------------------------------
    X_centered = X_centered - T(:,h)*P(:,h)';
    Y_centered = Y_centered - B(h)*T(:,h)*Q(:,h)';
end

% 预测 Prediction
% ------------------------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------------------------
% 现有另一组验证数据集X_validate,样本数量为r，特征量为仍为m，以以上构建的模型计算预测的Y
% 此处使用原始数据，故
num = 10;
r = n;
T_est = zeros(r,num);
X_validate = zeros(r,m);
Y_estimated = zeros(r,p);
% 对原始数据集居中列平均化
for j = 1:m
    % 注意，此处减去的平均值应该为模型数据集的平均值，而非新数据的平均值
    X_validate(1:r,j) =   X(1:r,j) - mean(X(:,j));
end

% 计算预测的T
for h = 1:num
    T_est(:,h) = X_validate*W(:,h);
    X_validate = X_validate - T_est(:,h)*P(:,h)';
end

% 计算预测的Y
for h = 1:num
    Y_estimated = Y_estimated + B(h)*T_est(:,h)*Q(:,h)';
end
for i = 1:p
   % 注意此处最终的输出需要加上数据集Y的均值
   Y_estimated(:,i) = Y_estimated(:,i) + mean(Y(:,i)); 
end

% ------------------------------------------------------------------------------------------------------------
% ------------------------------------------------------------------------------------------------------------
% 绘制SIMPLS预测的Y值与原始Y值对比
figure(2);
yfitPLS = [ones(n,1) X]*betaPLS;
scatter(Y(:,1), yfitPLS(:,1),80,'+','b');
hold on;
% PLSR NIPALS预测的数据与原始数据进行对比
scatter(Y(1:r,1), Y_estimated(1:r,1),80,'x','r');
legend('the SIMPLS algorithm', 'the NIPALS algorithm');

xlabel('the validate Y data');
ylabel('the estimated Y data');
%%

