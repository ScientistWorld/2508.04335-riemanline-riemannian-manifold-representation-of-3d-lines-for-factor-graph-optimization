/*
 * @Author: yanyan-li yanyan.li.camp@gmail.com
 * @Date: 2023-10-18 13:11:39
 * @LastEditTime: 2025-07-13 12:27:41
 * @LastEditors: yanyan-li yanyan.li.camp@gmail.com
 * @Description:
 * @FilePath: /RiemanLine/src/optimizer/factor/RiemanLineProjectionFactor.cc
 */

#include "RiemanLineProjectionFactor.hpp"
#include "src/optimizer/parametrization/line_parameterization.hpp"
#include "src/utils/UtilTransformer.hpp"
#include <Eigen/src/Core/Matrix.h>

Eigen::Matrix2d RiemanLineProjectionFactor::sqrt_info;
double RiemanLineProjectionFactor::sum_t;

Eigen::Matrix2d RiemanParaLineProjectionFactor::sqrt_info;
double RiemanParaLineProjectionFactor::sum_t;

Eigen::Matrix2d RiemanGlobalLocalLineProjectionFactor::sqrt_info;
double RiemanGlobalLocalLineProjectionFactor::sum_t;

Eigen::Matrix2d ZhaoLiangLineProjectionFactor::sqrt_info;
double ZhaoLiangLineProjectionFactor::sum_t; 


Eigen::Vector2d residual_from_riemanline_pose(const Eigen::Matrix3d &Rwc,
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

RiemanLineProjectionFactor::RiemanLineProjectionFactor(const Eigen::Vector4d &_obs_i)
    : obs_i(_obs_i){};

/*
  parameters[0]:  Twc
  parameters[1]:  line_orth
*/
bool RiemanLineProjectionFactor::Evaluate(double const *const *parameters,
                                          double *residuals,
                                          double **jacobians) const
{
  Eigen::Vector3d twc(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond qwc(parameters[0][6], parameters[0][3], parameters[0][4],
                         parameters[0][5]);
  qwc.normalize();
  Eigen::Matrix3d Rwc(qwc.toRotationMatrix());

  Vector6d line_w;
  line_w.head(3) = Eigen::Vector3d(parameters[1][0], parameters[1][1], parameters[1][2]);
  line_w.tail(3) = Eigen::Vector3d(parameters[1][3], parameters[1][4], parameters[1][5]);

  // Vector6d line_w(parameters[1][0], parameters[1][1], parameters[1][2], parameters[1][3], parameters[1][4], parameters[1][5]);

  // std::cout << Rwc << std::endl;
  Vector6d line_c = UtiliLine::plk_from_pose(line_w, Rwc, twc);

  // 直线的投影矩阵K为单位阵
  Eigen::Vector3d nc = line_c.head(3);
  // double l_norm = nc(0) * nc(0) + nc(1) * nc(1);
  double l_norm = nc.head<2>().squaredNorm();
  double l_sqrtnorm = sqrt(l_norm);
  // double l_trinorm = l_norm * l_sqrtnorm;

  double e1 = obs_i(0) * nc(0) + obs_i(1) * nc(1) + nc(2);
  double e2 = obs_i(2) * nc(0) + obs_i(3) * nc(1) + nc(2);
  Eigen::Map<Eigen::Vector2d> residual(residuals);
  residual(0) = e1 / l_sqrtnorm;
  residual(1) = e2 / l_sqrtnorm;

  // std::cout << "residual:" << residual(0) << ", " << residual(1) << std::endl;
  //    std::cout <<"---- sqrt_info: ------"<< sqrt_info << std::endl;
  sqrt_info.setIdentity();
  residual = sqrt_info * residual;

  if (jacobians)
  {
    const double eps = 1e-6;
    Eigen::Matrix<double, 2, 10> num_jacobian =
        Eigen::Matrix<double, 2, 10>::Zero();

    for (int k = 0; k < 10; k++)
    {
      Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
      Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3],
                            parameters[0][4], parameters[0][5]);
      Qi.normalize();

      Vector6d riemannian_line;
      riemannian_line.head(3) = Eigen::Vector3d(parameters[1][0], parameters[1][1], parameters[1][2]);
      riemannian_line.tail(3) = Eigen::Vector3d(parameters[1][3], parameters[1][4], parameters[1][5]);

      int a = k / 3, b = k % 3, c = k % 6;
      Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;
      Eigen::Matrix<double, 4, 1> delta_riemannian_bias;
      delta_riemannian_bias << 0.01 * eps, 0.01 * eps, 0.01 * eps, 0.01 * eps;

      Eigen::Matrix<double, 4, 1> delta_riemannian = Eigen::Matrix<double, 4, 1>(c == 0, c == 1, c == 2, c == 3) * eps;
      // std::cout << "c:" << k << "," << c << std::endl;

      if (a == 0)
        Pi += delta;
      else if (a == 1)
        Qi = Qi * UtiliLine::deltaQ(delta);
      else if (a == 2 || a == 3)
      { // line orth的前三个元素
        // if (jacobians[1])
        // Eigen::Vector4d theta_delta = Eigen::Vector4d::Zero();
        // theta_delta[a == 2 ? b : b + 1] = eps;
        riemannian_line = UtiliLine::deltaplk(riemannian_line, delta_riemannian + delta_riemannian_bias);
      }

      // Vector6d line_w_new = UtiliLine::ParaLine2PLK(PLK0, para_line_4, TangentBasisVectors);
      Eigen::Vector2d tmp_residual = residual_from_riemanline_pose(
          Qi.toRotationMatrix(), Pi, riemannian_line, obs_i);
      num_jacobian.col(k) = (tmp_residual - residual) / eps;
    }

    if (jacobians[0])
    {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose(
          jacobians[0]);
      jacobian_pose.leftCols<6>() = num_jacobian.leftCols<6>();
      jacobian_pose.rightCols<1>().setZero();
    }
    if (jacobians[1])
    {
      Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>
          jacobian_lineOrth_unique(jacobians[1]);
      jacobian_lineOrth_unique.leftCols<4>() = num_jacobian.rightCols<4>();
      jacobian_lineOrth_unique.rightCols<2>().setZero();
    }
  }
  return true;
}



RiemanParaLineProjectionFactor::RiemanParaLineProjectionFactor(
    const Eigen::Vector4d &_obs_i,
    const Vector6d &_PLK0)
    : obs_i(_obs_i), PLK0(_PLK0){};

/*
  parameters[0]:  Twc
  parameters[1]:  line_orth
*/
bool RiemanParaLineProjectionFactor::Evaluate(double const *const *parameters,
                                              double *residuals,
                                              double **jacobians) const
{
  Eigen::Vector3d twc(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond qwc(parameters[0][6], parameters[0][3], parameters[0][4],
                         parameters[0][5]);
  // global, local
  Eigen::Vector4d para_line_4(parameters[1][0], parameters[1][1],
                              parameters[2][0], parameters[2][1]);

  Vector6d line_w;
  double d = PLK0.head(3).norm();
  const double eps = 1e-6;
  Eigen::Matrix<double, 3, 2> TangentBasisVectors = UtiliLine::TangentBasis(PLK0.tail(3)); // for
  Eigen::Vector3d _manifold_direc_move = (parameters[1][0]) * TangentBasisVectors.col(0) + (parameters[1][1]) * TangentBasisVectors.col(1);
  Eigen::Vector3d _direct = PLK0.tail(3) * cos(_manifold_direc_move.norm()) + _manifold_direc_move / (_manifold_direc_move.norm()) * sin(_manifold_direc_move.norm());
  Eigen::Vector3d _normal = (PLK0.head(3).normalized() - _direct * (_direct.transpose() * PLK0.head(3).normalized())).normalized();
  Eigen::Vector3d TangentVector = UtiliLine::TangentVector(_direct, _normal); // for normal
  Eigen::Vector3d _manifold_normal_move = (parameters[2][0]) * TangentVector;
  // from TS to S
  _normal = _normal * cos(_manifold_normal_move.norm()) + _manifold_normal_move / (_manifold_normal_move.norm()) * sin(_manifold_normal_move.norm());
  // SO(2) update
  d += parameters[2][1];
  _normal = d * _normal;
  line_w[3] = _direct.x();
  line_w[4] = _direct.y();
  line_w[5] = _direct.z();
  line_w[0] = _normal.x();
  line_w[1] = _normal.y();
  line_w[2] = _normal.z();

  // std::cout << "factor:" << line_w << std::endl;

  Eigen::Matrix3d Rwc(qwc.toRotationMatrix());

  // std::cout << Rwc << std::endl;
  Vector6d line_c = UtiliLine::plk_from_pose(line_w, Rwc, twc);

  // 直线的投影矩阵K为单位阵
  Eigen::Vector3d nc = line_c.head(3);
  double l_norm = nc(0) * nc(0) + nc(1) * nc(1);
  double l_sqrtnorm = sqrt(l_norm);
  double l_trinorm = l_norm * l_sqrtnorm;

  double e1 = obs_i(0) * nc(0) + obs_i(1) * nc(1) + nc(2);
  double e2 = obs_i(2) * nc(0) + obs_i(3) * nc(1) + nc(2);
  Eigen::Map<Eigen::Vector2d> residual(residuals);
  residual(0) = e1 / l_sqrtnorm;
  residual(1) = e2 / l_sqrtnorm;

  //    std::cout <<"---- sqrt_info: ------"<< sqrt_info << std::endl;
  sqrt_info.setIdentity();
  residual = sqrt_info * residual;

  if (jacobians)
  {
    // const double eps = 1e-6;
    Eigen::Matrix<double, 2, 10> num_jacobian =
        Eigen::Matrix<double, 2, 10>::Zero();
    for (int k = 0; k < 10; k++)
    {
      Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
      Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3],
                            parameters[0][4], parameters[0][5]);

      Vector6d riemannian_line;
      // riemannian_line.head(3) = Eigen::Vector3d(parameters[1][0], parameters[1][1], parameters[1][2]);
      // riemannian_line.tail(3) = Eigen::Vector3d(parameters[2][0], parameters[2][1], parameters[2][2]);
      riemannian_line.head(3) = PLK0.head(3);
      riemannian_line.tail(3) = PLK0.tail(3);

      int a = k / 3, b = k % 3, c = k % 6;
      Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;
      Eigen::Matrix<double, 4, 1> delta_riemannian_0;
      delta_riemannian_0 << 0.01 * eps, 0.01 * eps, 0.01 * eps, 0.01 * eps;

      Eigen::Matrix<double, 4, 1> delta_riemannian = Eigen::Matrix<double, 4, 1>(c == 0, c == 1, c == 2, c == 3) * eps;
      // std::cout << "c:" << k << "," << c << std::endl;

      if (a == 0 && jacobians[0])
        Pi += delta;
      else if (a == 1 && jacobians[0])
        Qi = Qi * UtiliLine::deltaQ(delta);
      else if (a == 2 || a == 3)
      { // line orth的前三个元素
        if (jacobians[1] || jacobians[2])
          riemannian_line = UtiliLine::deltaplk(riemannian_line, delta_riemannian + delta_riemannian_0);
      }

      // Vector6d line_w_new = UtiliLine::ParaLine2PLK(PLK0, para_line_4, TangentBasisVectors);
      Eigen::Vector2d tmp_residual = residual_from_riemanline_pose(
          Qi.toRotationMatrix(), Pi, riemannian_line, obs_i);
      num_jacobian.col(k) = (tmp_residual - residual) / eps;
    }

    if (jacobians[0])
    {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose(
          jacobians[0]);
      jacobian_pose.leftCols<6>() = num_jacobian.leftCols<6>();
      jacobian_pose.rightCols<1>().setZero();
    }
    if (jacobians[1])
    {
      Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>>
          jacobian_lineOrth_shared(jacobians[1]);
      jacobian_lineOrth_shared.leftCols<2>() = num_jacobian.block(0, 6, 2, 2);
    }
    if (jacobians[2])
    {
      Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>>
          jacobian_lineOrth_unique(jacobians[2]);
      jacobian_lineOrth_unique.leftCols<2>() = num_jacobian.rightCols<2>();
    }
  }
  return true;
}

RiemanGlobalLocalLineProjectionFactor::RiemanGlobalLocalLineProjectionFactor(
    const Eigen::Vector4d &_obs_i)
    : obs_i(_obs_i){};

bool RiemanGlobalLocalLineProjectionFactor::Evaluate(double const *const *parameters,
                                                     double *residuals,
                                                     double **jacobians) const
{
  Eigen::Vector3d twc(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond qwc(parameters[0][6], parameters[0][3], parameters[0][4],
                         parameters[0][5]);

  Vector6d line_w;
  line_w.head(3) = Eigen::Vector3d(parameters[1][0], parameters[1][1], parameters[1][2]);
  line_w.tail(3) = Eigen::Vector3d(parameters[2][0], parameters[2][1], parameters[2][2]);

  // std::cout << "line_w:" << line_w << std::endl;
  // std::cout << "riemannian_line.head(3)" << line_w.head(3).transpose() * line_w.tail(3) << std::endl;
  Eigen::Matrix3d Rwc(qwc.toRotationMatrix());

  // std::cout << Rwc << std::endl;
  Vector6d line_c = UtiliLine::plk_from_pose(line_w, Rwc, twc);
  // std::cout << "times" << std::endl;
  // 直线的投影矩阵K为单位阵
  Eigen::Vector3d nc = line_c.head(3);
  double l_norm = nc(0) * nc(0) + nc(1) * nc(1);
  double l_sqrtnorm = sqrt(l_norm);
  double l_trinorm = l_norm * l_sqrtnorm;

  double e1 = obs_i(0) * nc(0) + obs_i(1) * nc(1) + nc(2);
  double e2 = obs_i(2) * nc(0) + obs_i(3) * nc(1) + nc(2);
  Eigen::Map<Eigen::Vector2d> residual(residuals);
  residual(0) = e1 / l_sqrtnorm;
  residual(1) = e2 / l_sqrtnorm;

  //    std::cout <<"---- sqrt_info: ------"<< sqrt_info << std::endl;
  sqrt_info.setIdentity();
  residual = sqrt_info * residual;

  if (jacobians)
  {
    const double eps = 1e-6;
    Eigen::Matrix<double, 2, 10> num_jacobian =
        Eigen::Matrix<double, 2, 10>::Zero();
    for (int k = 0; k < 10; k++)
    {
      Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
      Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3],
                            parameters[0][4], parameters[0][5]);

      Vector6d riemannian_line;
      riemannian_line.head(3) = Eigen::Vector3d(parameters[1][0], parameters[1][1], parameters[1][2]);
      riemannian_line.tail(3) = Eigen::Vector3d(parameters[2][0], parameters[2][1], parameters[2][2]);

      int a = k / 3, b = k % 3, c = k % 6;
      Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;
      Eigen::Matrix<double, 4, 1> delta_riemannian_0;
      delta_riemannian_0 << 0.01 * eps, 0.01 * eps, 0.01 * eps, 0.01 * eps;

      Eigen::Matrix<double, 4, 1> delta_riemannian = Eigen::Matrix<double, 4, 1>(c == 0, c == 1, c == 2, c == 3) * eps;
      // std::cout << "c:" << k << "," << c << std::endl;

      if (a == 0 && jacobians[0])
        Pi += delta;
      else if (a == 1 && jacobians[0])
        Qi = Qi * UtiliLine::deltaQ(delta);
      else if (a == 2 || a == 3)
      { // line orth的前三个元素
        if (jacobians[1] || jacobians[2])
          riemannian_line = UtiliLine::deltaplk(riemannian_line, delta_riemannian + delta_riemannian_0);
      }

      // Vector6d line_w_new = UtiliLine::ParaLine2PLK(PLK0, para_line_4, TangentBasisVectors);
      Eigen::Vector2d tmp_residual = residual_from_riemanline_pose(
          Qi.toRotationMatrix(), Pi, riemannian_line, obs_i);
      num_jacobian.col(k) = (tmp_residual - residual) / eps;
    }

    if (jacobians[0])
    {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose(
          jacobians[0]);
      jacobian_pose.leftCols<6>() = num_jacobian.leftCols<6>();
      jacobian_pose.rightCols<1>().setZero();
    }
    if (jacobians[1])
    {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>
          jacobian_lineOrth_unique(jacobians[1]);
      jacobian_lineOrth_unique.leftCols<2>() = num_jacobian.block(0, 6, 2, 2);
      // rightCols<2>();
      jacobian_lineOrth_unique.rightCols<1>().setZero();
    }
    if (jacobians[2])
    {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>
          jacobian_lineOrth_shared(jacobians[2]);
      jacobian_lineOrth_shared.leftCols<2>() = num_jacobian.rightCols<2>();
      jacobian_lineOrth_shared.rightCols<1>().setZero();
    }
  }
  return true;
}

Eigen::Vector2d residual_from_zhaoliang(const Eigen::Matrix3d &Rwci,
                                        const Eigen::Vector3d &twci,
                                        const Eigen::Matrix3d &Rwc0,
                                        const Eigen::Vector3d &twc0,
                                        const Eigen::Matrix3d &Rwc1,
                                        const Eigen::Vector3d &twc1,
                                        const Eigen::Vector4d &line_zhaoliang,
                                        const Eigen::Vector4d &obs_i)
{
  Eigen::Vector3d norm_0 = UtiliLine::Angle2Normvector(line_zhaoliang.head(2));
  Eigen::Vector3d norm_1 = UtiliLine::Angle2Normvector(line_zhaoliang.tail(2));

  Eigen::Vector3d norm_i = (twc1 - twci).dot(norm_1) * norm_0 - (twc0 - twci).dot(norm_0) * norm_1;

  norm_i = Rwci.transpose() * norm_i;

  double l_norm = norm_i(0) * norm_i(0) + norm_i(1) * norm_i(1);
  double l_sqrtnorm = sqrt(l_norm);

  double e1 = obs_i(0) * norm_i(0) + obs_i(1) * norm_i(1) + norm_i(2);
  double e2 = obs_i(2) * norm_i(0) + obs_i(3) * norm_i(1) + norm_i(2);

  Eigen::Vector2d residual;
  residual(0) = e1 / l_sqrtnorm;
  residual(1) = e2 / l_sqrtnorm;

  return residual;
}

ZhaoLiangLineProjectionFactor::ZhaoLiangLineProjectionFactor(const Eigen::Vector4d &_obs_i)
    : obs_i(_obs_i){};

/*
  parameters[0]:  Twc
  parameters[1]:  line_orth
*/
bool ZhaoLiangLineProjectionFactor::Evaluate(double const *const *parameters,
                                             double *residuals,
                                             double **jacobians) const
{
  Eigen::Vector3d twci(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond qwci(parameters[0][6], parameters[0][3], parameters[0][4],
                          parameters[0][5]);

  Eigen::Vector3d twc0(parameters[1][0], parameters[1][1], parameters[1][2]);
  Eigen::Quaterniond qwc0(parameters[1][6], parameters[1][3], parameters[1][4],
                          parameters[1][5]);

  Eigen::Vector3d twc1(parameters[2][0], parameters[2][1], parameters[2][2]);
  Eigen::Quaterniond qwc1(parameters[2][6], parameters[2][3], parameters[2][4],
                          parameters[2][5]);

  Eigen::Vector4d line_zhaoliang(parameters[3][0], parameters[3][1], parameters[3][2], parameters[3][3]);

  Eigen::Matrix3d Rwci(qwci.toRotationMatrix());
  Eigen::Matrix3d Rwc0(qwc0.toRotationMatrix());
  Eigen::Matrix3d Rwc1(qwc1.toRotationMatrix());

  Eigen::Map<Eigen::Vector2d> residual(residuals);
  residual = residual_from_zhaoliang(Rwci, twci, Rwc0, twc0, Rwc1, twc1, line_zhaoliang, obs_i);

  //    std::cout <<"---- sqrt_info: ------"<< sqrt_info << std::endl;
  sqrt_info.setIdentity();
  residual = sqrt_info * residual;

  if (jacobians)
  {
    const double eps = 1e-6;
    Eigen::Matrix<double, 2, 22> num_jacobian =
        Eigen::Matrix<double, 2, 22>::Zero();
    for (int k = 0; k < 22; k++)
    {
      Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
      Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3],
                            parameters[0][4], parameters[0][5]);
      Eigen::Vector3d P0(parameters[1][0], parameters[1][1], parameters[1][2]);
      Eigen::Quaterniond Q0(parameters[1][6], parameters[1][3],
                            parameters[1][4], parameters[1][5]);
      Eigen::Vector3d P1(parameters[2][0], parameters[2][1], parameters[2][2]);
      Eigen::Quaterniond Q1(parameters[2][6], parameters[2][3],
                            parameters[2][4], parameters[2][5]);

      Eigen::Vector4d ZhaoLinag_line(parameters[3][0], parameters[3][1], parameters[3][2], parameters[3][3]);

      Vector6d plk_line;
      plk_line.head(3) = Eigen::Vector3d(parameters[1][0], parameters[1][1], parameters[1][2]);
      plk_line.tail(3) = Eigen::Vector3d(parameters[1][3], parameters[1][4], parameters[1][5]);

      int a = k / 3, b = k % 3, c = k % 6;
      Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

      Eigen::Matrix<double, 4, 1> delta_zhaoliang = Eigen::Matrix<double, 4, 1>(c == 0, c == 1, c == 2, c == 3) * eps;

      if (a == 0 && jacobians[0])
        Pi += delta;
      else if (a == 1 && jacobians[0])
        Qi = Qi * UtiliLine::deltaQ(delta);
      else if (a == 2 && jacobians[1])
        P0 += delta;
      else if (a == 3 && jacobians[1])
        Q0 = Q0 * UtiliLine::deltaQ(delta);
      else if (a == 4 && jacobians[2])
        P1 += delta;
      else if (a == 5 && jacobians[2])
        Q1 = Q1 * UtiliLine::deltaQ(delta);
      else if (a == 6 || a == 7)
      { // line orth的前三个元素
        if (jacobians[3])
          ZhaoLinag_line = ZhaoLinag_line + delta_zhaoliang;
      }

      // Vector6d line_w_new = UtiliLine::ParaLine2PLK(PLK0, para_line_4, TangentBasisVectors);
      Eigen::Vector2d tmp_residual = residual_from_zhaoliang(
          Qi.toRotationMatrix(), Pi, Q0.toRotationMatrix(), P0, Q1.toRotationMatrix(), P1, ZhaoLinag_line, obs_i);
      num_jacobian.col(k) = (tmp_residual - residual) / eps;
    }

    if (jacobians[0])
    {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose(
          jacobians[0]);
      jacobian_pose.leftCols<6>() = num_jacobian.leftCols<6>();
      jacobian_pose.rightCols<1>().setZero();
    }
    if (jacobians[1])
    {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>
          jacobian_pose0(jacobians[1]);
      jacobian_pose0.leftCols<6>() = num_jacobian.block(0, 6, 2, 6);
      jacobian_pose0.rightCols<1>().setZero();
    }
    if (jacobians[2])
    {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>
          jacobian_pose1(jacobians[2]);
      jacobian_pose1.leftCols<6>() = num_jacobian.block(0, 12, 2, 6);
      jacobian_pose1.rightCols<1>().setZero();
    }
    if (jacobians[3])
    {
      Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>>
          jacobian_lineOrth_unique(jacobians[3]);
      jacobian_lineOrth_unique = num_jacobian.rightCols<4>();
    }
  }
  return true;
}