/*
 * @Author: yanyan-li yanyan.li.camp@gmail.com
 * @Date: 2025-07-22 10:27:08
 * @LastEditTime: 2025-07-22 10:49:13
 * @LastEditors: yanyan-li yanyan.li.camp@gmail.com
 * @Description:
 * @FilePath: /RiemanLine/examples/_test_paraline.cc
 */
#include "src/optimizer/parametrization/line_parameterization.hpp"
#include "src/utils/UtilTransformer.hpp"
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <iomanip>
#include <iostream>
#include <random>

// 假设你已经有 ParaRiemanLineParameterization 类实现

void CheckJacobianNumerically(const ParaRiemanLineParameterization &param, int num_lines)
{
    const int global_dim = param.GlobalSize();
    const int local_dim = param.LocalSize();
    const double eps = 1e-8;

    Eigen::VectorXd x(global_dim);
    x.setZero();
    x[2] = 1.0; // z-axis

    for (int i = 0; i < num_lines; ++i)
    {
        Eigen::Vector3d n = Eigen::Vector3d::Random().normalized();
        x.segment<3>(3 + 3 * i) = n;
    }

    Eigen::MatrixXd J_analytic(global_dim, local_dim);
    param.ComputeJacobian(x.data(), J_analytic.data());

    Eigen::MatrixXd J_numeric(global_dim, local_dim);
    for (int i = 0; i < local_dim; ++i)
    {
        Eigen::VectorXd delta = Eigen::VectorXd::Zero(local_dim);
        delta[i] = eps;

        Eigen::VectorXd x_plus(global_dim);
        param.Plus(x.data(), delta.data(), x_plus.data());

        J_numeric.col(i) = (x_plus - x) / eps;
    }

    std::cout << "\n[Jacobian Check] Analytic vs Numeric:\n";
    std::cout << "Max diff: " << (J_analytic - J_numeric).cwiseAbs().maxCoeff() << "\n";
    std::cout << "Norm diff: " << (J_analytic - J_numeric).norm() << "\n";

    if ((J_analytic - J_numeric).norm() < 1e-6)
        std::cout << "✅ Jacobian matches numerically.\n";
    else
        std::cout << "❌ Jacobian mismatch. Check your implementation.\n";
}

void TestParaRiemanLineParameterization(int num_lines = 3)
{
    ParaRiemanLineParameterization param(num_lines);

    const int global_dim = param.GlobalSize();
    const int local_dim = param.LocalSize();

    Eigen::VectorXd x(global_dim);
    Eigen::VectorXd x_plus(global_dim);
    Eigen::VectorXd delta(local_dim);

    // 1. 初始化方向向量 d
    Eigen::Vector3d d(0.0, 0.0, 1.0); // 朝 Z 轴
    x.segment<3>(0) = d;

    // 2. 初始化 normal 向量
    for (int i = 0; i < num_lines; ++i)
    {
        Eigen::Vector3d n = Eigen::Vector3d::Random().normalized();
        double scale = 1.0 + 0.1 * i;
        x.segment<3>(3 + 3 * i) = scale * n;
    }

    // 3. 构造微小扰动 delta
    delta.setRandom();
    delta *= 1e-4;

    // 4. Plus 运算
    assert(param.Plus(x.data(), delta.data(), x_plus.data()));

    // 5. 计算 Jacobian
    Eigen::MatrixXd J(global_dim, local_dim);
    assert(param.ComputeJacobian(x.data(), J.data()));

    // 6. 线性近似结果
    Eigen::VectorXd approx = x + J * delta;

    // 7. 打印比较结果
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "===== Testing ParaRiemanLineParameterization =====\n";
    std::cout << "Original x:       " << x.transpose() << "\n";
    std::cout << "Delta:            " << delta.transpose() << "\n";
    std::cout << "Plus(x,delta):    " << x_plus.transpose() << "\n";
    std::cout << "x + J*delta:      " << approx.transpose() << "\n";
    std::cout << "Error norm:       " << (x_plus - approx).norm() << "\n";

    if ((x_plus - approx).norm() < 1e-6)
    {
        std::cout << "✅ PASS: Jacobian matches Plus() up to 1e-6\n";
    }
    else
    {
        std::cout << "❌ FAIL: Jacobian does not match Plus(). Consider checking parameterization.\n";
    }
}

int main()
{
    int num_lines = 3;
    ParaRiemanLineParameterization param(num_lines);

    CheckJacobianNumerically(param, num_lines);
    return 0;
}