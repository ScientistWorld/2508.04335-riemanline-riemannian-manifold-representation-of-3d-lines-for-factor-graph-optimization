/*
 * @Author: yanyan-li yanyan.li.camp@gmail.com
 * @Date: 2023-11-21 12:31:55
 * @LastEditTime: 2025-07-22 02:38:37
 * @LastEditors: yanyan-li yanyan.li.camp@gmail.com
 * @Description:
 * @FilePath: /RiemanLine/src/optimizer/factor/ParaRiemanlineProjectionFactor.cc
 */

#include "ParaRiemanlineProjectionFactor.hpp"
#include "src/optimizer/parametrization/line_parameterization.hpp"
#include "src/utils/UtilTransformer.hpp"
#include <Eigen/src/Core/Matrix.h>

Eigen::Vector2d residual_from_parariemanline_pose(const Eigen::Matrix3d &Rwc,
                                                  const Eigen::Vector3d &twc,
                                                  const Vector6d &line_w,
                                                  const Eigen::Vector4d &obs_i)
{
  Vector6d line_c = UtiliLine::plk_from_pose(line_w, Rwc, twc);

  // 直线的投影矩阵K为单位阵
  Eigen::Vector3d nc = line_c.head(3);
  double l_norm = nc(0) * nc(0) + nc(1) * nc(1);
  double l_sqrtnorm = sqrt(l_norm);
  // double l_trinorm = l_norm * l_sqrtnorm;

  double e1 = obs_i(0) * nc(0) + obs_i(1) * nc(1) + nc(2);
  double e2 = obs_i(2) * nc(0) + obs_i(3) * nc(1) + nc(2);

  Eigen::Vector2d residual;
  residual(0) = e1 / l_sqrtnorm;
  residual(1) = e2 / l_sqrtnorm;

  return residual;
}

RiemanianlinesetProjectionFactorX::RiemanianlinesetProjectionFactorX(
    const Eigen::Vector4d &obs, int position_idx, int num_lines_)
    : obs_i(obs), position(position_idx), num_lines(num_lines_)
{
  set_num_residuals(2);
  mutable_parameter_block_sizes()->push_back(7);                 // pose
  mutable_parameter_block_sizes()->push_back(3 + 3 * num_lines); // direction + per-line normals
}

bool RiemanianlinesetProjectionFactorX::Evaluate(double const *const *parameters,
                                                 double *residuals,
                                                 double **jacobians) const
{
  using Vec6 = Eigen::Matrix<double, 6, 1>;

  // Pose
  Eigen::Vector3d t_wc(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond q_wc(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
  q_wc.normalize();
  Eigen::Matrix3d R_wc = q_wc.toRotationMatrix();

  // Line direction & normal
  const double *line_params = parameters[1];
  Eigen::Vector3d d(line_params[0], line_params[1], line_params[2]);
  Eigen::Vector3d n(line_params[3 + 3 * position + 0],
                    line_params[3 + 3 * position + 1],
                    line_params[3 + 3 * position + 2]);

  Vec6 line_w;
  line_w.head<3>() = n;
  line_w.tail<3>() = d;

  Vec6 line_c = UtiliLine::plk_from_pose(line_w, R_wc, t_wc);
  Eigen::Vector3d nc = line_c.head<3>();

  double norm = nc.head<2>().squaredNorm();
  double sqrt_norm = std::sqrt(norm);

  Eigen::Map<Eigen::Vector2d> res(residuals);
  res(0) = (obs_i.head<2>().dot(nc.head<2>()) + nc(2)) / sqrt_norm;
  res(1) = (obs_i.tail<2>().dot(nc.head<2>()) + nc(2)) / sqrt_norm;

  if (jacobians)
  {
    // === Jacobian w.r.t pose ===
    if (jacobians[0])
    {
      // numerical diff w.r.t pose (translation + rotation)
      const double eps = 1e-6;
      Eigen::Matrix<double, 2, 6> J_pose;

      for (int i = 0; i < 6; ++i)
      {
        Eigen::Vector3d t_eps = t_wc;
        Eigen::Quaterniond q_eps = q_wc;

        if (i < 3)
        {
          t_eps[i] += eps;
        }
        else
        {
          Eigen::Vector3d delta_rot = Eigen::Vector3d::Zero();
          delta_rot[i - 3] = eps;
          q_eps = q_eps * UtiliLine::deltaQ(delta_rot);
        }
        // q_eps.normalize();

        Vec6 line_c_eps = UtiliLine::plk_from_pose(line_w, q_eps.toRotationMatrix(), t_eps);
        Eigen::Vector3d nc_eps = line_c_eps.head<3>();
        double sqrt_norm_eps = std::sqrt(nc_eps.head<2>().squaredNorm());

        Eigen::Vector2d res_eps;
        res_eps(0) = (obs_i.head<2>().dot(nc_eps.head<2>()) + nc_eps(2)) / sqrt_norm_eps;
        res_eps(1) = (obs_i.tail<2>().dot(nc_eps.head<2>()) + nc_eps(2)) / sqrt_norm_eps;

        J_pose.col(i) = (res_eps - res) / eps;
      }

      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_out(jacobians[0]);
      J_out.leftCols<6>() = J_pose;
      J_out.rightCols<1>().setZero();
    }

    if (jacobians[1])
    {
      const double eps = 1e-6;
      const int total_dim = 3 + 3 * num_lines;

      Eigen::Matrix<double, 2, Eigen::Dynamic> J_line(2, total_dim);
      J_line.setZero();

      // Only perturb d[0:2] and current line's n_i
      for (int i = 0; i < 3; ++i) // shared direction
      {
        Eigen::VectorXd line_eps = Eigen::Map<const Eigen::VectorXd>(line_params, total_dim);
        line_eps[i] += eps;

        Eigen::Vector3d d_eps(line_eps[0], line_eps[1], line_eps[2]);
        Eigen::Vector3d n_eps = line_eps.segment<3>(3 + 3 * position);

        Vec6 line_w_eps;
        line_w_eps.head<3>() = n_eps;
        line_w_eps.tail<3>() = d_eps;

        Vec6 line_c_eps = UtiliLine::plk_from_pose(line_w_eps, R_wc, t_wc);
        Eigen::Vector3d nc_eps = line_c_eps.head<3>();
        double sqrt_norm_eps = std::sqrt(nc_eps.head<2>().squaredNorm());

        Eigen::Vector2d res_eps;
        res_eps(0) = (obs_i.head<2>().dot(nc_eps.head<2>()) + nc_eps(2)) / sqrt_norm_eps;
        res_eps(1) = (obs_i.tail<2>().dot(nc_eps.head<2>()) + nc_eps(2)) / sqrt_norm_eps;

        J_line.col(i) = (res_eps - res) / eps;
      }

      // now perturb normal for this line only
      for (int i = 0; i < 3; ++i)
      {
        int idx = 3 + 3 * position + i;
        Eigen::VectorXd line_eps = Eigen::Map<const Eigen::VectorXd>(line_params, total_dim);
        line_eps[idx] += eps;

        Eigen::Vector3d d_eps(line_eps[0], line_eps[1], line_eps[2]);
        Eigen::Vector3d n_eps = line_eps.segment<3>(3 + 3 * position);

        Vec6 line_w_eps;
        line_w_eps.head<3>() = n_eps;
        line_w_eps.tail<3>() = d_eps;

        Vec6 line_c_eps = UtiliLine::plk_from_pose(line_w_eps, R_wc, t_wc);
        Eigen::Vector3d nc_eps = line_c_eps.head<3>();
        double sqrt_norm_eps = std::sqrt(nc_eps.head<2>().squaredNorm());

        Eigen::Vector2d res_eps;
        res_eps(0) = (obs_i.head<2>().dot(nc_eps.head<2>()) + nc_eps(2)) / sqrt_norm_eps;
        res_eps(1) = (obs_i.tail<2>().dot(nc_eps.head<2>()) + nc_eps(2)) / sqrt_norm_eps;

        J_line.col(idx) = (res_eps - res) / eps;
      }

      Eigen::Map<Eigen::Matrix<double, 2, Eigen::Dynamic, Eigen::RowMajor>> J_out(jacobians[1], 2, total_dim);
      J_out = J_line;
    }
  }

  return true;
}