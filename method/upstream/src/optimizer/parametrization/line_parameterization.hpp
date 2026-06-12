/***
 * @Author: yanyan-li yanyan.li.camp@gmail.com
 * @Date: 2023-11-06 01:49:59
 * @LastEditTime: 2023-11-20 08:07:43
 * @LastEditors: yanyan-li yanyan.li.camp@gmail.com
 * @Description:
 * @FilePath: /LineRepresentation/src/optimizer/parametrization/line_parameterization.hpp
 */

#pragma once

#include "../../utils/UtilParameterss.hpp"
#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>

class LineOrthParameterization : public ceres::LocalParameterization
{
  virtual bool Plus(const double *x,
                    const double *delta,
                    double *x_plus_delta) const;
  virtual bool ComputeJacobian(const double *x, double *jacobian) const;
  virtual int GlobalSize() const { return 4; };
  virtual int LocalSize() const { return 4; };
};

class ParaLineOrthParameterization : public ceres::LocalParameterization
{
  virtual bool Plus(const double *x,
                    const double *delta,
                    double *x_plus_delta) const;
  virtual bool ComputeJacobian(const double *x, double *jacobian) const;
  virtual int GlobalSize() const { return 2; };
  virtual int LocalSize() const { return 2; };
};

class LineZhaoLiangParameterization : public ceres::LocalParameterization
{
  virtual bool Plus(const double *x,
                    const double *delta,
                    double *x_plus_delta) const;
  virtual bool ComputeJacobian(const double *x, double *jacobian) const;
  virtual int GlobalSize() const { return 6; };
  virtual int LocalSize() const { return 4; };
};

class LineRiemanParameterization : public ceres::LocalParameterization
{ // LineRiemanParameterization(int global,int local);
  virtual bool Plus(const double *x,
                    const double *delta,
                    double *x_plus_delta) const;
  virtual bool ComputeJacobian(const double *x, double *jacobian) const;
  virtual int GlobalSize() const { return 6; };
  virtual int LocalSize() const { return 4; };
};

/**
 * @brief ParaRiemanLineParameterization
 * parametrization for a set of RiemanLine
 *
 */
class ParaRiemanLineParameterization : public ceres::LocalParameterization
{
public:
  explicit ParaRiemanLineParameterization(int num_lines) : num_lines_(num_lines) {}

  virtual bool Plus(const double *x,
                    const double *delta,
                    double *x_plus_delta) const;
  virtual bool ComputeJacobian(const double *x, double *jacobian) const;
  virtual int GlobalSize() const override
  {
    return 3 + 3 * num_lines_; // 3 for direction, 3N for anchors
  };

  virtual int LocalSize() const override
  {
    return 2 + 2 * num_lines_; // 2 for direction update, 2N for anchors
  };

private:
  int num_lines_;
};

class LineParaRiemanLineParameterization : public ceres::LocalParameterization
{
public:
  explicit LineParaRiemanLineParameterization(int num_lines) : num_lines_(num_lines) {}

  virtual bool Plus(const double *x,
                    const double *delta,
                    double *x_plus_delta) const;
  virtual bool ComputeJacobian(const double *x, double *jacobian) const;
  // virtual int GlobalSize() const { return global_size_vec; };
  // virtual int LocalSize() const { return local_size_vec; };
  virtual int GlobalSize() const override
  {
    return 3 + 3 * num_lines_; // 3 for direction, 3N for anchors
  };

  virtual int LocalSize() const override
  {
    return 2 + 2 * num_lines_; // 2 for direction update, 2N for anchors
  };

private:
  int num_lines_;
};

class LineRiemanParameterizationGlobal : public ceres::LocalParameterization
{
  virtual bool Plus(const double *x,
                    const double *delta,
                    double *x_plus_delta) const;
  virtual bool ComputeJacobian(const double *x, double *jacobian) const;
  virtual int GlobalSize() const { return 3; };
  virtual int LocalSize() const { return 2; };
};

class LineRiemanParameterizationUnique : public ceres::LocalParameterization
{
public:
  LineRiemanParameterizationUnique(const double *vanishing_direction);
  const double *vd_;
  virtual bool Plus(const double *x,
                    const double *delta,
                    double *x_plus_delta) const;
  virtual bool ComputeJacobian(const double *x, double *jacobian) const;
  virtual int GlobalSize() const { return 3; };
  virtual int LocalSize() const { return 2; };
};
