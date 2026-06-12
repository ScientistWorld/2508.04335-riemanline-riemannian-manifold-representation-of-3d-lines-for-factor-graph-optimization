/*** 
 * @Author: yanyan-li yanyan.li.camp@gmail.com
 * @Date: 2023-11-06 01:49:59
 * @LastEditTime: 2023-11-21 13:44:02
 * @LastEditors: yanyan-li yanyan.li.camp@gmail.com
 * @Description: 
 * @FilePath: /LineRepresentation/src/optimizer/factor/RiemanLineProjectionFactor.hpp
 */

#pragma once

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include "src/utils/UtilTransformer.hpp"

typedef Eigen::Matrix<double, 6, 1> Vector6d;
class RiemanLineProjectionFactor : public ceres::SizedCostFunction<2, 7, 6>
{
public:
    RiemanLineProjectionFactor(const Eigen::Vector4d &_pts_i);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    void check(double **parameters);

    Eigen::Vector4d obs_i;
    static Eigen::Matrix2d sqrt_info;
    static double sum_t;
};


class RiemanGlobalLocalLineProjectionFactor : public ceres::SizedCostFunction<2, 7, 3, 3>
{
public:
    RiemanGlobalLocalLineProjectionFactor(const Eigen::Vector4d &_pts_i);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    void check(double **parameters);

    Eigen::Vector4d obs_i;
    static Eigen::Matrix2d sqrt_info;
    static double sum_t;
};

class RiemanParaLineProjectionFactor : public ceres::SizedCostFunction<2, 7, 2, 2>
{
public:
    RiemanParaLineProjectionFactor(const Eigen::Vector4d &_pts_i, const Vector6d &_PLK0);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    void check(double **parameters);

    Eigen::Vector4d obs_i;
    static Eigen::Matrix2d sqrt_info;
    static double sum_t;

    Vector6d PLK0;
    // Eigen::Matrix<double, 3, 2> TangentBasisVectors;
    Eigen::Vector3d TangentBaseVector;
};


class ZhaoLiangLineProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 4>
{
public:
    ZhaoLiangLineProjectionFactor(const Eigen::Vector4d &_pts_i);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    void check(double **parameters);

    Eigen::Vector4d obs_i;
    static Eigen::Matrix2d sqrt_info;
    static double sum_t;
};




template <typename T>
    Eigen::Matrix<T, 3, 1> AngleToNormvector(const Eigen::Matrix<T, 2, 1>& angle) {
        Eigen::Matrix<T, 3, 1> norm;
        norm << ceres::cos(angle(0)) * ceres::cos(angle(1)),
                ceres::sin(angle(0)) * ceres::cos(angle(1)),
                ceres::sin(angle(1));
        return norm;
    }

template <typename T>
Eigen::Matrix<T, 2, 1> residual_from_zhaoliangLine(const Eigen::Matrix<T, 3, 3> &Rwci,
                                                const Eigen::Matrix<T, 3, 1> &twci,
                                                const Eigen::Matrix<T, 3, 3> &Rwc0,
                                                const Eigen::Matrix<T, 3, 1> &twc0,
                                                const Eigen::Matrix<T, 3, 3> &Rwc1,
                                                const Eigen::Matrix<T, 3, 1> &twc1,
                                                const Eigen::Matrix<T, 4, 1> &line_zhaoliang,
                                                const Eigen::Matrix<T, 4, 1> &obs_i_T)
{
  Eigen::Matrix<T, 2, 1> angle0 = line_zhaoliang.template head<2>().eval();
  Eigen::Matrix<T, 2, 1> angle1 = line_zhaoliang.template tail<2>().eval();

  Eigen::Matrix<T, 3, 1> norm_0 = AngleToNormvector(angle0);
  Eigen::Matrix<T, 3, 1> norm_1 = AngleToNormvector(angle1);

  Eigen::Matrix<T, 3, 1> norm_i = (twc1 - twci).dot(norm_1) * norm_0 - (twc0 - twci).dot(norm_0) * norm_1;

  norm_i = Rwci.transpose() * norm_i;

  T l_norm = norm_i(0) * norm_i(0) + norm_i(1) * norm_i(1);
  T l_sqrtnorm = ceres::sqrt(l_norm);

  T e1 = obs_i_T(0) * norm_i(0) + obs_i_T(1) * norm_i(1) + norm_i(2);
  T e2 = obs_i_T(2) * norm_i(0) + obs_i_T(3) * norm_i(1) + norm_i(2);

  Eigen::Matrix<T, 2, 1> residual;
  residual(0) = e1 / l_sqrtnorm;
  residual(1) = e2 / l_sqrtnorm;

  return residual;
}


struct ZhaoLiangLineProjectionFactorAutoDiff
{
    const Eigen::Vector4d obs_i_double;
    const Eigen::Matrix2d sqrt_info_double;

    ZhaoLiangLineProjectionFactorAutoDiff(const Eigen::Vector4d& obs_i_val, const Eigen::Matrix2d& sqrt_info_val)
        : obs_i_double(obs_i_val), sqrt_info_double(sqrt_info_val) {}

    template <typename T>
    bool operator()(const T* const pose_i_params_ptr,
                    const T* const pose_0_params_ptr,
                    const T* const pose_1_params_ptr,
                    const T* const line_zhaoliang_ptr,
                    T* residuals_ptr) const
    {
        
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> twci_map(pose_i_params_ptr);
        Eigen::Quaternion<T> qwci(pose_i_params_ptr[6], pose_i_params_ptr[3], pose_i_params_ptr[4], pose_i_params_ptr[5]);

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> twc0_map(pose_0_params_ptr);
        Eigen::Quaternion<T> qwc0(pose_0_params_ptr[6], pose_0_params_ptr[3], pose_0_params_ptr[4], pose_0_params_ptr[5]);

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> twc1_map(pose_1_params_ptr);
        Eigen::Quaternion<T> qwc1(pose_1_params_ptr[6], pose_1_params_ptr[3], pose_1_params_ptr[4], pose_1_params_ptr[5]);

        Eigen::Map<const Eigen::Matrix<T, 4, 1>> line_zhaoliang_map(line_zhaoliang_ptr);

        Eigen::Matrix<T, 3, 3> Rwci = qwci.normalized().toRotationMatrix();
        Eigen::Matrix<T, 3, 3> Rwc0 = qwc0.normalized().toRotationMatrix();
        Eigen::Matrix<T, 3, 3> Rwc1 = qwc1.normalized().toRotationMatrix();

        Eigen::Matrix<T, 4, 1> obs_i_T = obs_i_double.template cast<T>();

        Eigen::Matrix<T, 3, 1> twci = twci_map;
        Eigen::Matrix<T, 3, 1> twc0 = twc0_map;
        Eigen::Matrix<T, 3, 1> twc1 = twc1_map;
        Eigen::Matrix<T, 4, 1> line_zhaoliang = line_zhaoliang_map;

        Eigen::Matrix<T, 2, 1> raw_residual = residual_from_zhaoliangLine(
            Rwci, twci, Rwc0, twc0, Rwc1, twc1, line_zhaoliang, obs_i_T);

        Eigen::Map<Eigen::Matrix<T, 2, 1>> weighted_residual(residuals_ptr);
        weighted_residual = sqrt_info_double.template cast<T>() * raw_residual;

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector4d& obs_i_val, const Eigen::Matrix2d& sqrt_info_val)
    {
        return new ceres::AutoDiffCostFunction<ZhaoLiangLineProjectionFactorAutoDiff, 2, 7, 7, 7, 4>(
            new ZhaoLiangLineProjectionFactorAutoDiff(obs_i_val, sqrt_info_val));
    }
};