clear;
clc;
%%
% 验证PCR回归，与Matlab官方库对比，并用于手势协同分析实验
%%
% 加载matlab官方数据集'spectra'
load spectra
% 定义X数据集为原始数据集
X = NIR;
Y = octane;
[n,m] = size(X);
[n,p] = size(Y);
%%
% 调用官方函数直接计算主成份，投影后的数据集以及协方差矩阵的特征值
% 使用10个主成分
[PCALoadings,PCAScores,PCAVar] = pca(X,'Economy',false);
betaPCR = regress(Y-mean(Y), PCAScores(:,1:10));
betaPCR = PCALoadings(:,1:10)*betaPCR;
betaPCR = [mean(Y) - mean(X)*betaPCR; betaPCR];
yfitPCR = [ones(n,1) X]*betaPCR;

%%
% 自定义PCA代码并与官方函数库对照-NIPALS方法
X_centered = X;
Y_centered = Y;
% 对原始数据集进行列居中处理
for j = 1:m
    X_centered(:,j) =   X(:,j) - mean(X(:,j));
end
for j = 1:p
    Y_centered(:,j) =   Y(:,j) - mean(Y(:,j));
end
% 使用matlab计算协方差矩阵的特征向量和特征值组成的对角矩阵
[V,D] = eig(X_centered'*X_centered);
% 定义迭代的残差矩阵X_iteration
X_iteration = X_centered;
% 定义保留的主成分数，score和loading矩阵，即投影后的矩阵和主成分矩阵(由特征向量组成)
num = 10;
T = zeros(n,num);
P = zeros(m,num);
eigenValues = zeros(num,1);
% 设置迭代容忍度和存储前一步T(:,h)的临时变量
tolerance = 0.000001;
t_temp = zeros(n,1);
% 设置迭代的游标，或者说是PLS成份序号
h = 1;
% ------------------------------------------------------------------------------------------------------------
% 迭代得到num个成份
for h = 1:num
    % step(1)
    % ---------------------------------------------------------------------
    % 取T(:,h)为任意一个X_centered中的列向量，此处直接取第一列
    T(:,h) = X_iteration(:,1);

    % step(2) to step(5)
    % 迭代直到收敛到容忍度内的主成分
    while(1)
        P(:,h) = X_iteration'*T(:,h)/(T(:,h)'*T(:,h));
        % 归一化P(:,h)
        P(:,h) =  P(:,h)/sqrt(P(:,h)'*P(:,h));
        t_temp = T(:,h);
        T(:,h) = X_iteration*P(:,h)/(P(:,h)'*P(:,h));

        % 检查当前T(:,h)与上一步T(:,h)是否相等以决定是否继续迭代
        if max(abs(T(:,h)-t_temp)) <= tolerance
            % 存储按顺序排列的特征值
            eigenValues(h) = P(:,h)'*(X_centered'*X_centered)*P(:,h);
            break;
        else
        end
    end
    % 计算残差，更新数据矩阵
    % ---------------------------------------------------------------------
    X_iteration = X_iteration - T(:,h)*P(:,h)';
end
%%
% 定义测试集样本的数量
r = n;
% 将原始数据降维到主成分空间(T)后，使用OLS最小二乘回归获取系数矩阵
B_inPca = inv(T'*T)*T'*Y_centered;
%B_inPca = regress(Y-mean(Y), T(:,1:num));
% 将系数矩阵从主成分空间转化到原始空间
B_estimated = P*B_inPca;

% 定义测试集，此处直接使用原始数据的前r行
X_validate = zeros(r,m);
% 对原始数据集居中列平均化
for j = 1:m
    % 注意，此处减去的平均值应该为模型数据集的平均值，而非新数据的平均值
    X_validate(:,j) =   X(1:r,j) - mean(X(:,j));
end

Y_estimated = X_validate*B_estimated;
for i = 1:p
   % 注意此处最终的输出需要加上数据集Y的均值
   Y_estimated(:,i) = Y_estimated(:,i) + mean(Y(:,i)); 
end
%%
% 绘制原始数据与预测的数据对比
figure(1);
scatter(Y(:,1), yfitPCR(:,1),80,'+','b');
hold on;
% PLSR NIPALS预测的数据与原始数据进行对比
scatter(Y(1:r,1), Y_estimated(1:r,1),80,'x','r');
legend('the matlab algorithm', 'the self-defined algorithm');

%%




