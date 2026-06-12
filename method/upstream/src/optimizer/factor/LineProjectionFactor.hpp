/***
 * @Author: yanyan-li yanyan.li.camp@gmail.com
 * @Date: 2023-03-05 00:39:11
 * @LastEditTime: 2023-08-04 18:41:46
 * @LastEditors: yanyan-li yanyan.li.camp@gmail.com
 * @Description:
 * @FilePath: /JointFactorGraph/src/optimizer/factor/LineProjectionFactor.hpp
 */
#pragma once

#include "src/utils/UtilParameterss.hpp"
#include <Eigen/Dense>
#include <ceres/ceres.h>

typedef Eigen::Matrix<double, 6, 1> Vector6d;
class PointProjectionFactor : public ceres::SizedCostFunction<2, 7, 3>
{
public:
  PointProjectionFactor(const Eigen::Vector2d &_pts_i);
  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
  void check(double **parameters);

  Eigen::Vector2d obs_i;
  static Eigen::Matrix2d sqrt_info;
  static double sum_t;
};

class LineProjectionFactor : public ceres::SizedCostFunction<2, 7, 4>
{
public:
  LineProjectionFactor(const Eigen::Vector4d &_pts_i, double weight = 1.0);

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

  void check(double **parameters);

  Eigen::Vector4d obs_i;
  Eigen::Matrix2d sqrt_info; // Make this per-instance
  double weight;
};

// class LineProjectionFactor : public ceres::SizedCostFunction<2, 7, 4>
// {
// public:
//   LineProjectionFactor(const Eigen::Vector4d &_pts_i);
//   virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
//   void check(double **parameters);

//   Eigen::Vector4d obs_i;
//   static Eigen::Matrix2d sqrt_info;
//   static double sum_t;
// };

class ParallelLineFactor : public ceres::SizedCostFunction<1, 4, 4>
{
public:
  ParallelLineFactor() {}

  virtual bool Evaluate(double const *const *parameters,
                        double *residuals,
                        double **jacobians) const;
};

class ParallelRiemanLineFactor : public ceres::SizedCostFunction<1, 6, 6>
{
public:
  ParallelRiemanLineFactor() {}

  virtual bool Evaluate(double const *const *parameters,
                        double *residuals,
                        double **jacobians) const;
};

class ParaLineProjectionFactor : public ceres::SizedCostFunction<2, 7, 2, 2>
{
public:
  ParaLineProjectionFactor(const Eigen::Vector4d &_pts_i, const Vector6d &_PLK0, const Eigen::Matrix<double, 3, 2> &_TangentBasisVectors);
  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
  void check(double **parameters);

  Eigen::Vector4d obs_i;
  Vector6d PLK0;
  Eigen::Matrix<double, 3, 2> TangentBasisVectors;
  static Eigen::Matrix2d sqrt_info;
  static double sum_t;
};

class ParaLineTranslationFactor : public ceres::SizedCostFunction<2, 7, 2, 2>
{
public:
  ParaLineTranslationFactor(const Eigen::Vector4d &_pts_i, const Vector6d &_PLK0, const Eigen::Matrix<double, 3, 2> &_TangentBasisVectors);
  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
  void check(double **parameters);

  Eigen::Vector4d obs_i;
  Vector6d PLK0;
  Eigen::Matrix<double, 3, 2> TangentBasisVectors;
  static Eigen::Matrix2d sqrt_info;
  static double sum_t;
};

class ParaLineRotationFactor : public ceres::SizedCostFunction<1, 7, 2>
{
public:
  ParaLineRotationFactor(const Eigen::Vector4d &_pts_i, const Vector6d &_PLK0, const Eigen::Matrix<double, 3, 2> &_TangentBasisVectors);
  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
  void check(double **parameters);

  Eigen::Vector4d obs_i;
  Vector6d PLK0;
  Eigen::Matrix<double, 3, 2> TangentBasisVectors;
  static Eigen::Matrix2d sqrt_info;
  static double sum_t;
};

class RiemanianlinesetRotationFactor : public ceres::SizedCostFunction<1, 7, global_size_vec>
{
public:
  RiemanianlinesetRotationFactor(const Eigen::Vector4d &_pts_i);
  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
  void check(double **parameters);

  Eigen::Vector4d obs_i;
  static Eigen::Matrix2d sqrt_info;
  static double sum_t;
};

// class RiemanianlinesetRotationFactor2 : public ceres::SizedCostFunction<1, 7, global_size_vec_16>
// {
// public:
//   RiemanianlinesetRotationFactor2(const Eigen::Vector4d &_pts_i);
//   virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
//   void check(double **parameters);

//   Eigen::Vector4d obs_i;
//   static Eigen::Matrix2d sqrt_info;
//   static double sum_t;
// };

// class RiemanianlinesetRotationFactor3 : public ceres::SizedCostFunction<1, 7, global_size_vec_n3>
// {
// public:
//   RiemanianlinesetRotationFactor3(const Eigen::Vector4d &_pts_i);
//   virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
//   void check(double **parameters);

//   Eigen::Vector4d obs_i;
//   static Eigen::Matrix2d sqrt_info;
//   static double sum_t;
// };