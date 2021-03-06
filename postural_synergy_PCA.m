clear;
clc;
%%
% 验证PCA，与Matlab官方库对比，并用于手势协同分析实验
%%
% 加载matlab官方数据集'hald',为13x4的数据矩阵
load hald;
% 调用官方函数直接计算主成份，投影后的数据集以及协方差矩阵的特征值
[coeff,score,latent] = pca(ingredients);

%%
% 自定义PCA代码并与官方函数库对照-NIPALS方法

% 定义X数据集为原始数据集
X = ingredients;
[n,m] = size(X);
X_centered = X;
% 对原始数据集进行列居中处理
for j = 1:m
    X_centered(:,j) =   X(:,j) - mean(X(:,j));
end
% 使用matlab计算协方差矩阵的特征向量和特征值组成的对角矩阵
[V,D] = eig(X_centered'*X_centered);
% 定义迭代的残差矩阵X_iteration
X_iteration = X_centered;
% 定义保留的主成分数，score和loading矩阵，即投影后的矩阵和主成分矩阵(由特征向量组成)
num = m;
T = zeros(n,num);
P = zeros(m,num);
eigenValues = zeros(num,1);
% 设置迭代容忍度和存储前一步T(:,h)的临时变量
tolerance = 0.000001;
t_temp = zeros(n,1);
% 设置迭代的游标，或者说是PLS成份序号
h = 1;

% ------------------------------------------------------------------------------------------------------------
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
            % 注意此处的特征值为协方差矩阵的特征值，而matlab PCA方法使用的为散布矩阵(离散度矩阵)，故后者的特征值为前者的(n-1)倍
            eigenValues(h) = P(:,h)'*(X_centered'*X_centered)*P(:,h);
            break;
        else
        end
    end
    
    % 计算残差，更新数据矩阵
    % ---------------------------------------------------------------------
    X_iteration = X_iteration - T(:,h)*P(:,h)';
end
 
 
 
 
 
 
 
 
 
 
 
 