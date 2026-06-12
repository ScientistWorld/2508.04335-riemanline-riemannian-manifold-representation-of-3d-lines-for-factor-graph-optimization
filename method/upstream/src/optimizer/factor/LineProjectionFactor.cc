#include "LineProjectionFactor.hpp"
#include "src/optimizer/parametrization/line_parameterization.hpp"
#include "src/utils/UtilTransformer.hpp"
#include <Eigen/src/Core/Matrix.h>

Eigen::Matrix2d PointProjectionFactor::sqrt_info;
double PointProjectionFactor::sum_t;

PointProjectionFactor::PointProjectionFactor(const Eigen::Vector2d &_obs_i)
    : obs_i(_obs_i) {};

/*
  parameters[0]:  Twc
  parameters[1]:  Point3d
*/
bool PointProjectionFactor::Evaluate(double const *const *parameters,
                                     double *residuals,
                                     double **jacobians) const
{
  Eigen::Vector3d twc(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond qwc(parameters[0][6], parameters[0][3], parameters[0][4],
                         parameters[0][5]);

  Eigen::Vector3d point_3d(parameters[1][0], parameters[1][1],
                           parameters[1][2]);

  Eigen::Matrix3d Rwc(qwc.toRotationMatrix());
  Eigen::Vector3d point_c = qwc.inverse() * (point_3d - twc);

  Eigen::Map<Eigen::Vector2d> residual(residuals);

  double dep_j = point_c.z();
  residual = (point_c / dep_j).head<2>() - obs_i;

  sqrt_info.setIdentity();
  residual = sqrt_info * residual;
  if (false)
  {
    if (jacobians)
    {
      Eigen::Matrix<double, 2, 3> reduce(2, 3);

      reduce << 1. / point_c(2), 0, -point_c(0) / (point_c(2) * point_c(2)), 0,
          1. / point_c(2), -point_c(1) / (point_c(2) * point_c(2));

      if (jacobians[0])
      {
        // std::cout <<"jacobian_ex_pose"<<"\n";
        Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose(
            jacobians[0]);

        Eigen::Matrix<double, 3, 6> jaco_pintc_pose;
        jaco_pintc_pose.setZero();
        jaco_pintc_pose.block(0, 0, 3, 3) = -Rwc.inverse(); // Lc_t
        jaco_pintc_pose.block(0, 3, 3, 3) =
            UtiliLine::skew_symmetric(point_3d - twc);

        jacobian_pose.leftCols<6>() = reduce * jaco_pintc_pose;
        jacobian_pose.rightCols<1>().setZero();
      }
      if (jacobians[1])
      {
        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jacobian_Point(
            jacobians[1]);

        jacobian_Point = reduce * Rwc.inverse();
      }
    }
  }
  // check jacobian
  // std::cout
  //     << "---------- check jacobian ----------\n";
  // if (jacobians[0])
  //   std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(
  //                    jacobians[0])
  //             << std::endl
  //             << std::endl;
  // if (jacobians[1])
  //   std::cout << Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>(
  //                    jacobians[1])
  //             << std::endl
  //             << std::endl;

  if (jacobians)
  {
    const double eps = 1e-6;
    Eigen::Matrix<double, 2, 9> num_jacobian;
    for (int k = 0; k < 9; k++)
    {
      Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
      Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4],
                            parameters[0][5]);

      Eigen::Vector3d point_3d(parameters[1][0], parameters[1][1],
                               parameters[1][2]);

      int a = k / 3, b = k % 3;
      Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

      if (a == 0)
        Pi += delta;
      else if (a == 1)
        Qi = Qi * UtiliLine::deltaQ(delta);
      else if (a == 2)
        point_3d += delta;

      Eigen::Vector3d point_c = Qi.inverse() * (point_3d - Pi);

      double dep_j = point_c.z();

      Eigen::Vector2d tmp_residual;
      tmp_residual =
          (point_c / dep_j).head<2>() - obs_i; // 误差 = 坐标重投影误差
      tmp_residual = sqrt_info * tmp_residual;

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
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jacobian_point(jacobians[1]);
      jacobian_point = num_jacobian.rightCols<3>(); //.rightCols<2>();
    }

    // std::cout << "num_jacobian pose:\n"
    //           << num_jacobian.block(0, 0, 2, 6) << "\n"
    //           << std::endl;

    // std::cout << "num_jacobian point:\n"
    //           << num_jacobian.block(0, 6, 2, 3) << "\n"
    //           << std::endl;
  }

  return true;
}

// Eigen::Matrix2d LineProjectionFactor::sqrt_info;
// double LineProjectionFactor::sum_t;

LineProjectionFactor::LineProjectionFactor(const Eigen::Vector4d &_obs_i, double _weight)
    : obs_i(_obs_i),  weight(_weight)
{
  // sqrt_info = sqrt(weight) * Eigen::Matrix2d::Identity();
  sqrt_info = Eigen::Matrix2d::Identity() * weight;
}

/*
  parameters[0]:  Twc
  parameters[1]:  line_orth
*/
// LineProjectionFactor::Evaluate with analytical Jacobians
bool LineProjectionFactor::Evaluate(double const *const *parameters,
                                    double *residuals,
                                    double **jacobians) const
{
  
  // // Pose
  Vector3d t(parameters[0][0], parameters[0][1], parameters[0][2]);
  Quaterniond q(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
  Matrix3d R = q.normalized().toRotationMatrix();

  // Line orthonormal representation
  Vector4d orth = Map<const Vector4d>(parameters[1]);
  Vector6d line_w = UtiliLine::orth_to_plk(orth);
  Vector6d line_c = UtiliLine::plk_from_pose(line_w, R, t);

  Vector3d nc = line_c.head<3>();
  double norm = nc.head<2>().squaredNorm();
  double sqrt_norm = sqrt(norm);
  double cube_norm = norm * sqrt_norm;

  double e1 = obs_i(0) * nc(0) + obs_i(1) * nc(1) + nc(2);
  double e2 = obs_i(2) * nc(0) + obs_i(3) * nc(1) + nc(2);

  Map<Vector2d> residual(residuals);
  residual(0) = sqrt(weight) * e1 / sqrt_norm;
  residual(1) = sqrt(weight) * e2 / sqrt_norm;

  if (jacobians)
  {
    const double eps = 1e-6;
    Eigen::Matrix<double, 2, 10> num_jacobian;  // 6 pose + 4 line_orth

    Eigen::Vector2d base_residual;  // store original residual
    {
      // unpack pose
      Eigen::Vector3d t(parameters[0][0], parameters[0][1], parameters[0][2]);
      Eigen::Quaterniond q(parameters[0][6], parameters[0][3],
                          parameters[0][4], parameters[0][5]);
      q.normalize();
      Eigen::Matrix3d R = q.toRotationMatrix();

      // unpack line orthonormal parameters
      Eigen::Vector4d orth(parameters[1][0], parameters[1][1],
                          parameters[1][2], parameters[1][3]);
      Vector6d line_w = UtiliLine::orth_to_plk(orth);
      Vector6d line_c = UtiliLine::plk_from_pose(line_w, R, t);

      Eigen::Vector3d nc = line_c.head<3>();
      double norm = nc.head<2>().squaredNorm();
      double sqrt_norm = sqrt(norm);
      double e1 = obs_i(0) * nc(0) + obs_i(1) * nc(1) + nc(2);
      double e2 = obs_i(2) * nc(0) + obs_i(3) * nc(1) + nc(2);

      base_residual(0) = sqrt(weight) * e1 / sqrt_norm;
      base_residual(1) = sqrt(weight) * e2 / sqrt_norm;
    }

    for (int k = 0; k < 10; ++k)
    {
      // clone input
      Eigen::Vector3d t(parameters[0][0], parameters[0][1], parameters[0][2]);
      Eigen::Quaterniond q(parameters[0][6], parameters[0][3],
                          parameters[0][4], parameters[0][5]);
      q.normalize();
      Eigen::Vector4d orth(parameters[1][0], parameters[1][1],
                          parameters[1][2], parameters[1][3]);

      // perturbation delta
      int block = k / 3;
      int index = k % 3;
      Eigen::Vector3d delta = Eigen::Vector3d::Zero();
      delta[index] = eps;

      if (k < 3)             // pose translation
        t += delta;
      else if (k < 6)        // pose rotation
        q = q * UtiliLine::deltaQ(delta);
      else                   // line orth parameters
        orth[k - 6] += eps;

      q.normalize();
      Eigen::Matrix3d R = q.toRotationMatrix();
      Vector6d line_w = UtiliLine::orth_to_plk(orth);
      Vector6d line_c = UtiliLine::plk_from_pose(line_w, R, t);

      Eigen::Vector3d nc = line_c.head<3>();
      double norm = nc.head<2>().squaredNorm();
      double sqrt_norm = sqrt(norm);
      double e1 = obs_i(0) * nc(0) + obs_i(1) * nc(1) + nc(2);
      double e2 = obs_i(2) * nc(0) + obs_i(3) * nc(1) + nc(2);

      Eigen::Vector2d tmp_residual;
      tmp_residual(0) = sqrt(weight) * e1 / sqrt_norm;
      tmp_residual(1) = sqrt(weight) * e2 / sqrt_norm;

      num_jacobian.col(k) = (tmp_residual - base_residual) / eps;
    }

    // assign back
    if (jacobians[0])
    {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_pose(jacobians[0]);
      J_pose.leftCols<6>() = num_jacobian.leftCols<6>();
      J_pose.rightCols<1>().setZero();
    }

    if (jacobians[1])
    {
      Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> J_line(jacobians[1]);
      J_line = num_jacobian.rightCols<4>();
    }
  }
  return true;
}

bool ParallelLineFactor::Evaluate(double const *const *parameters,
                                  double *residuals,
                                  double **jacobians) const
{
  Eigen::Map<const Eigen::Vector4d> line1(parameters[0]);
  Eigen::Map<const Eigen::Vector4d> line2(parameters[1]);

  Eigen::Matrix<double, 6, 1> plk1 = UtiliLine::orth_to_plk(line1);
  Eigen::Matrix<double, 6, 1> plk2 = UtiliLine::orth_to_plk(line2);

  Eigen::Vector3d v1 = plk1.tail<3>().normalized();
  Eigen::Vector3d v2 = plk2.tail<3>().normalized();

  double dot = v1.dot(v2);
  dot = std::min(1.0, std::max(-1.0, dot)); // 防止数值精度溢出
  double angle = std::acos(dot);            // in radians

  // -----> 核心：自适应权重
  double weight = std::min(10.0, 1.0 + 20.0 * angle); // angle 越大，weight 越大

  residuals[0] = weight * (1.0 - dot);

  if (jacobians)
  {
    const double eps = 1e-6;

    if (jacobians[0])
    {
      Eigen::Map<Eigen::Matrix<double, 1, 4>> J_line1(jacobians[0]);
      for (int i = 0; i < 4; ++i)
      {
        Eigen::Vector4d line1_plus = line1;
        line1_plus(i) += eps;
        Eigen::Matrix<double, 6, 1> plk1_plus = UtiliLine::orth_to_plk(line1_plus);
        Eigen::Vector3d v1_plus = plk1_plus.tail<3>().normalized();

        double dot_plus = v1_plus.dot(v2);
        dot_plus = std::min(1.0, std::max(-1.0, dot_plus));
        double angle_plus = std::acos(dot_plus);
        double weight_plus = std::min(10.0, 1.0 + 20.0 * angle_plus);
        double residual_plus = weight_plus * (1.0 - dot_plus);

        J_line1(0, i) = (residual_plus - residuals[0]) / eps;
      }
    }

    if (jacobians[1])
    {
      Eigen::Map<Eigen::Matrix<double, 1, 4>> J_line2(jacobians[1]);
      for (int i = 0; i < 4; ++i)
      {
        Eigen::Vector4d line2_plus = line2;
        line2_plus(i) += eps;
        Eigen::Matrix<double, 6, 1> plk2_plus = UtiliLine::orth_to_plk(line2_plus);
        Eigen::Vector3d v2_plus = plk2_plus.tail<3>().normalized();

        double dot_plus = v1.dot(v2_plus);
        dot_plus = std::min(1.0, std::max(-1.0, dot_plus));
        double angle_plus = std::acos(dot_plus);
        double weight_plus = std::min(10.0, 1.0 + 20.0 * angle_plus);
        double residual_plus = weight_plus * (1.0 - dot_plus);

        J_line2(0, i) = (residual_plus - residuals[0]) / eps;
      }
    }
  }

  return true;
  // // Extract orthonormal parameters
  // Eigen::Map<const Eigen::Vector4d> line1(parameters[0]);
  // Eigen::Map<const Eigen::Vector4d> line2(parameters[1]);

  // // Convert both to Plücker
  // Eigen::Matrix<double, 6, 1> plk1 = UtiliLine::orth_to_plk(line1);
  // Eigen::Matrix<double, 6, 1> plk2 = UtiliLine::orth_to_plk(line2);

  // Eigen::Vector3d v1 = plk1.tail<3>().normalized();
  // Eigen::Vector3d v2 = plk2.tail<3>().normalized();

  // // Residual: angular difference between directions
  // residuals[0] = 1.0 - v1.dot(v2); // = 0 when parallel

  // // if (jacobians)
  // // {
  // //   // Optionally use numeric differentiation
  // // }
  // if (jacobians)
  // {
  //   // Finite difference for line1
  //   if (jacobians[0])
  //   {
  //     Eigen::Map<Eigen::Matrix<double, 1, 4>> J_line1(jacobians[0]);
  //     const double eps = 1e-6;

  //     for (int i = 0; i < 4; ++i)
  //     {
  //       Eigen::Vector4d line1_plus = line1;
  //       line1_plus(i) += eps;
  //       Eigen::Matrix<double, 6, 1> plk_plus = UtiliLine::orth_to_plk(line1_plus);
  //       Eigen::Vector3d v1_plus = plk_plus.tail<3>().normalized();

  //       double r_plus = 1.0 - v1_plus.dot(v2);

  //       J_line1(0, i) = (r_plus - residuals[0]) / eps;
  //     }
  //   }

  //   // Finite difference for line2
  //   if (jacobians[1])
  //   {
  //     Eigen::Map<Eigen::Matrix<double, 1, 4>> J_line2(jacobians[1]);
  //     const double eps = 1e-6;

  //     for (int i = 0; i < 4; ++i)
  //     {
  //       Eigen::Vector4d line2_plus = line2;
  //       line2_plus(i) += eps;
  //       Eigen::Matrix<double, 6, 1> plk_plus = UtiliLine::orth_to_plk(line2_plus);
  //       Eigen::Vector3d v2_plus = plk_plus.tail<3>().normalized();

  //       double r_plus = 1.0 - v1.dot(v2_plus);

  //       J_line2(0, i) = (r_plus - residuals[0]) / eps;
  //     }
  //   }
  // }

  // return true;
}

bool ParallelRiemanLineFactor::Evaluate(double const *const *parameters,
                                        double *residuals,
                                        double **jacobians) const
{
  // using namespace Eigen;

  Map<const Vector6d> plk1(parameters[0]);
  Map<const Vector6d> plk2(parameters[1]);

  Vector3d v1 = plk1.tail<3>().normalized();
  Vector3d v2 = plk2.tail<3>().normalized();

  double dot = v1.dot(v2);
  dot = std::clamp(dot, -1.0, 1.0);
  double angle = std::acos(dot);
  double weight = std::min(10.0, 1.0 + 20.0 * angle);

  residuals[0] = weight * (1.0 - dot);

  if (jacobians)
  {
    const double eps = 1e-6;

    if (jacobians[0])
    {
      Map<Matrix<double, 1, 6>> J_line1(jacobians[0]);
      for (int i = 0; i < 6; ++i)
      {
        Vector6d plk1_plus = plk1;
        plk1_plus(i) += eps;
        Vector3d v1_plus = plk1_plus.tail<3>().normalized();

        double dot_plus = v1_plus.dot(v2);
        dot_plus = std::clamp(dot_plus, -1.0, 1.0);
        double angle_plus = std::acos(dot_plus);
        double weight_plus = std::min(10.0, 1.0 + 20.0 * angle_plus);
        double residual_plus = weight_plus * (1.0 - dot_plus);

        J_line1(0, i) = (residual_plus - residuals[0]) / eps;
      }
    }

    if (jacobians[1])
    {
      Map<Matrix<double, 1, 6>> J_line2(jacobians[1]);
      for (int i = 0; i < 6; ++i)
      {
        Vector6d plk2_plus = plk2;
        plk2_plus(i) += eps;
        Vector3d v2_plus = plk2_plus.tail<3>().normalized();

        double dot_plus = v1.dot(v2_plus);
        dot_plus = std::clamp(dot_plus, -1.0, 1.0);
        double angle_plus = std::acos(dot_plus);
        double weight_plus = std::min(10.0, 1.0 + 20.0 * angle_plus);
        double residual_plus = weight_plus * (1.0 - dot_plus);

        J_line2(0, i) = (residual_plus - residuals[0]) / eps;
      }
    }
  }

  return true;
}

Eigen::Vector2d residual_from_paraline_pose(const Eigen::Matrix3d &Rwc,
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

Eigen::Matrix2d ParaLineProjectionFactor::sqrt_info;
double ParaLineProjectionFactor::sum_t;

ParaLineProjectionFactor::ParaLineProjectionFactor(
    const Eigen::Vector4d &_obs_i, const Vector6d &_PLK0, const Eigen::Matrix<double, 3, 2> &_TangentBasisVectors)
    : obs_i(_obs_i), PLK0(_PLK0), TangentBasisVectors(_TangentBasisVectors) {};

/*
  parameters[0]:  Twc
  parameters[1]:  line_orth
*/
bool ParaLineProjectionFactor::Evaluate(double const *const *parameters,
                                        double *residuals,
                                        double **jacobians) const
{
  Eigen::Map<Eigen::Vector2d> residual(residuals);
  {
    Eigen::Vector3d twc(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond qwc(parameters[0][6], parameters[0][3], parameters[0][4],
                           parameters[0][5]);

    Eigen::Vector4d para_line_4(parameters[1][0], parameters[1][1],
                                parameters[2][0], parameters[2][1]);
    // std::cout<<"para_line_4:"<< para_line_4<<std::endl;

    // Vector3d LinePlaneNormal = PLK0.head(3);
    // Vector3d LineDirection = PLK0.tail(3);
    // double direction_move0 = para_line_4(0);
    // double direction_move1 = para_line_4(1);
    // Eigen::Vector3d direction_move = direction_move0 * TangentBasisVectors.col(0) + direction_move1 * TangentBasisVectors.col(1);
    // LineDirection = (LineDirection + direction_move).normalized();
    // LinePlaneNormal = (LinePlaneNormal - LineDirection * (LineDirection.transpose() * LinePlaneNormal)).normalized();

    // Vector6d plk_new;
    // plk_new.head(3) = LinePlaneNormal;
    // plk_new.tail(3) = LineDirection_new;
    // PLK0(plk_new);

    Vector6d line_w = UtiliLine::ParaLine2PLK(PLK0, para_line_4, TangentBasisVectors);
    residual = residual_from_paraline_pose(qwc.toRotationMatrix(), twc,
                                           line_w, obs_i);

    // std::cout<<"residual:"<<residual<<std::endl;

    sqrt_info.setIdentity();
    residual = sqrt_info * residual;
  }

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

      Eigen::Vector4d para_line_4(parameters[1][0], parameters[1][1],
                                  parameters[2][0], parameters[2][1]);

      int a = k / 3, b = k % 3;
      Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

      if (a == 0 && jacobians[0])
        Pi += delta;
      else if (a == 1 && jacobians[0])
        Qi = Qi * UtiliLine::deltaQ(delta);
      else if (a == 2)
      { // line orth的前三个元素
        if (b == 0 || b == 1)
        {
          if (jacobians[1])
            para_line_4.head(2) += Eigen::Vector2d(delta.x(), delta.y());
        }
        else if (b == 2)
        {
          if (jacobians[2])
            para_line_4[2] += delta.z();
        }
      }
      else if (a == 3 && jacobians[2])
      { // line orth的最后一个元素
        para_line_4[3] += delta.x();
      }

      Vector6d line_w_new = UtiliLine::ParaLine2PLK(PLK0, para_line_4, TangentBasisVectors);
      Eigen::Vector2d tmp_residual = residual_from_paraline_pose(
          Qi.toRotationMatrix(), Pi, line_w_new, obs_i);
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
      jacobian_lineOrth_shared = num_jacobian.block(0, 6, 2, 2); //.rightCols<2>();
    }
    if (jacobians[2])
    {
      Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>>
          jacobian_lineOrth_unique(jacobians[2]);
      jacobian_lineOrth_unique = num_jacobian.rightCols<2>();
    }
  }

  // update PLK0;

  return true;
}

Eigen::Matrix2d ParaLineTranslationFactor::sqrt_info;
double ParaLineTranslationFactor::sum_t;

ParaLineTranslationFactor::ParaLineTranslationFactor(
    const Eigen::Vector4d &_obs_i, const Vector6d &_PLK0, const Eigen::Matrix<double, 3, 2> &_TangentBasisVectors)
    : obs_i(_obs_i), PLK0(_PLK0), TangentBasisVectors(_TangentBasisVectors) {};

/*
  parameters[0]:  Twc
  parameters[1]:  line_orth
*/
bool ParaLineTranslationFactor::Evaluate(double const *const *parameters,
                                         double *residuals,
                                         double **jacobians) const
{
  Eigen::Map<Eigen::Vector2d> residual(residuals);
  {
    Eigen::Vector3d twc(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond qwc(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector4d para_line_4(parameters[1][0], parameters[1][1],
                                parameters[2][0], parameters[2][1]);
    // std::cout<<"para_line_4:"<< para_line_4<<std::endl;

    Vector6d line_w = UtiliLine::ParaLine2PLK(PLK0, para_line_4, TangentBasisVectors);
    residual = residual_from_paraline_pose(qwc.toRotationMatrix(), twc,
                                           line_w, obs_i);

    // std::cout<<"residual:"<<residual<<std::endl;

    sqrt_info.setIdentity();
    residual = sqrt_info * residual;
  }

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

      Eigen::Vector4d para_line_4(parameters[1][0], parameters[1][1],
                                  parameters[2][0], parameters[2][1]);

      int a = k / 3, b = k % 3;
      Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

      if (a == 0 && jacobians[0])
        Pi += delta;
      else if (a == 1 && jacobians[0])
        Qi = Qi; // * UtiliLine::deltaQ(delta);
      else if (a == 2)
      { // line orth的前三个元素
        if (b == 0 || b == 1)
        {
          if (jacobians[1])
            para_line_4.head(3) += delta;
        }
        else if (b == 2)
        {
          if (jacobians[2])
            para_line_4.head(3) += delta;
        }
      }
      else if (a == 3 && jacobians[2])
      { // line orth的最后一个元素
        para_line_4[3] += delta.x();
      }

      Vector6d line_w_new = UtiliLine::ParaLine2PLK(PLK0, para_line_4, TangentBasisVectors);
      Eigen::Vector2d tmp_residual = residual_from_paraline_pose(
          Qi.toRotationMatrix(), Pi, line_w_new, obs_i);
      num_jacobian.col(k) = (tmp_residual - residual) / eps;
    }

    if (jacobians[0])
    {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose(
          jacobians[0]);
      jacobian_pose.leftCols<3>() = num_jacobian.leftCols<3>();
      jacobian_pose.rightCols<4>().setZero();
    }
    if (jacobians[1])
    {
      Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>>
          jacobian_lineOrth_shared(jacobians[1]);
      jacobian_lineOrth_shared = num_jacobian.block(0, 6, 2, 2); //.rightCols<2>();
    }
    if (jacobians[2])
    {
      Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>>
          jacobian_lineOrth_unique(jacobians[2]);
      jacobian_lineOrth_unique = num_jacobian.rightCols<2>();
    }
  }
  return true;
}

double residual_from_paraline_rotation(const Eigen::Matrix3d &Rwc,
                                       const Eigen::Vector2d &line_d_move,
                                       const Vector6d &plk,
                                       const Eigen::Matrix<double, 3, 2> &TangentBasisVectors,
                                       const Eigen::Vector4d &obs_i)
{

  // Eigen::Vector3d LinePlaneNormal = plk.head(3).normalized();
  Eigen::Vector3d LineDirection = plk.tail(3).normalized();

  double direction_move0 = line_d_move(0);
  double direction_move1 = line_d_move(1);

  Eigen::Vector3d direction_move = direction_move0 * TangentBasisVectors.col(0) + direction_move1 * TangentBasisVectors.col(1);
  Eigen::Vector3d LineDirection_new = (LineDirection + direction_move).normalized();

  // todo:
  //  Eigen::Vector3d obs_line_direction_norm = E igen::Vector3d(obs_i(0)* obs_depth(0), obs_i(1)* obs_depth(0), obs_depth(0))\
    //                                            - Eigen::Vector3d(obs_i(2)*obs_depth(1), obs_i(3)*obs_depth(1), obs_depth(1));

  Eigen::Vector3d obs_line_plane_norm = Eigen::Vector3d(obs_i(0), obs_i(1), 1).cross(Eigen::Vector3d(obs_i(2), obs_i(3), 1));

  Eigen::Vector3d obs_line_plane_norm_w = Rwc * obs_line_plane_norm;
  obs_line_plane_norm_w = obs_line_plane_norm_w.normalized();

  // Eigen::Vector3d obs_line_direction_norm_w = Rwc*obs_line_direction_norm.normalized();
  // obs_line_direction_norm_w = obs_line_direction_norm_w.normalized();

  //    std::cout<<"obs_line_d:"<<obs_line_d(0)<<","<<obs_line_d(1)<<","<<obs_line_d(2)<<std::endl;
  //    std::cout<<"obs_line_w:"<<obs_line_w(0)<<","<<obs_line_w(1)<<","<<obs_line_w(2)<<std::endl;
  //    std::cout<<"LineDirection_new:"<<LineDirection_new(0)<<","<<LineDirection_new(1)<<","<<LineDirection_new(2)<<std::endl;

  double residual = 0.1 * obs_line_plane_norm_w.transpose() * LineDirection_new;
  // std::cout << "obs_line_d:" << residual << "," << std::endl;

  return residual;
}

Eigen::Matrix2d ParaLineRotationFactor::sqrt_info;
double ParaLineRotationFactor::sum_t;

// ParaLineRotationFactor::ParaLineRotationFactor(
//         const Eigen::Vector4d& _obs_i, const Eigen::Vector2d& _obs_depth, const Vector6d& _PLK0, const Eigen::Matrix<double, 3, 2>& _TangentBasisVectors)
//         : obs_i(_obs_i), PLK0(_PLK0), TangentBasisVectors(_TangentBasisVectors), obs_depth(_obs_depth){};
ParaLineRotationFactor::ParaLineRotationFactor(
    const Eigen::Vector4d &_obs_i, const Vector6d &_PLK0, const Eigen::Matrix<double, 3, 2> &_TangentBasisVectors)
    : obs_i(_obs_i), PLK0(_PLK0), TangentBasisVectors(_TangentBasisVectors) {};

/*
  parameters[0]:  Twc
  parameters[1]:  line_orth
*/
bool ParaLineRotationFactor::Evaluate(double const *const *parameters,
                                      double *residuals,
                                      double **jacobians) const
{

  {
    Eigen::Vector3d twc(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond qwc(parameters[0][6], parameters[0][3], parameters[0][4],
                           parameters[0][5]);
    Eigen::Vector2d para_line_2(parameters[1][0], parameters[1][1]);
    // std::cout<<"para_line_4:"<< para_line_4<<std::endl;
    (*residuals) = residual_from_paraline_rotation(qwc.toRotationMatrix(), para_line_2, PLK0, TangentBasisVectors, obs_i);
    // std::cout<<"residual:"<<residual<<std::endl;
    sqrt_info.setIdentity();
  }
  if (jacobians)
  {
    const double eps = 1e-6;
    Eigen::Matrix<double, 1, 8> num_jacobian =
        Eigen::Matrix<double, 1, 8>::Zero();
    for (int k = 0; k < 8; k++)
    {
      Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
      Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3],
                            parameters[0][4], parameters[0][5]);

      Eigen::Vector2d para_line_2(parameters[1][0], parameters[1][1]);
      int a = k / 3, b = k % 3;
      Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;
      // Eigen::Vector3d delta_x = Eigen::Vector3d(0, 0, 0);
      //      if (a == 0 && jacobians[0])
      //          //Pi += delta_x;
      //  int x = 0;
      //      // Pi += Eigen::Vector3d(0,0,0);
      //      else
      if (a == 1 && jacobians[0])
        Qi = Qi * UtiliLine::deltaQ(delta);
      else if (a == 2 && jacobians[1])
      {
        if (b == 0)
          para_line_2(0) += eps;
        if (b == 1)
          para_line_2(1) += eps;
      }
      double tmp_residual = residual_from_paraline_rotation(Qi.toRotationMatrix(), para_line_2, PLK0, TangentBasisVectors, obs_i);

      double residual = (*residuals);
      num_jacobian(0, k) = (tmp_residual - residual) / eps;
    }

    if (jacobians[0])
    {
      Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose(
          jacobians[0]);
      jacobian_pose.leftCols<3>().setZero();
      jacobian_pose.block(0, 3, 1, 3) = num_jacobian.block(0, 3, 1, 3); // leftCols<6>();
      // jacobian_pose.leftCols<6>() = num_jacobian.leftCols<6>();
      jacobian_pose.rightCols<1>().setZero();
    }
    if (jacobians[1])
    {
      Eigen::Map<Eigen::Matrix<double, 1, 2, Eigen::RowMajor>>
          jacobian_lineOrth_shared(jacobians[1]);
      jacobian_lineOrth_shared = num_jacobian.rightCols<2>();
    }
  }
  return true;
}

double residual_from_riemannian_rotation(const Eigen::Matrix3d &Rwc,
                                         const Vector3d &LineDirection,
                                         const Eigen::Vector4d &obs_i)
{

  Eigen::Vector3d obs_line_plane_norm = Eigen::Vector3d(obs_i(0), obs_i(1), 1).cross(Eigen::Vector3d(obs_i(2), obs_i(3), 1));

  Eigen::Vector3d obs_line_plane_norm_w = Rwc * obs_line_plane_norm;
  obs_line_plane_norm_w = obs_line_plane_norm_w.normalized();

  double residual = 0.2 * obs_line_plane_norm_w.transpose() * LineDirection;
  // std::cout << "obs_line_d:" << residual << "," << std::endl;

  return residual;
}
Eigen::Matrix2d RiemanianlinesetRotationFactor::sqrt_info;
double RiemanianlinesetRotationFactor::sum_t;
RiemanianlinesetRotationFactor::RiemanianlinesetRotationFactor(const Eigen::Vector4d &_pts_i) : obs_i(_pts_i) {};

bool RiemanianlinesetRotationFactor::Evaluate(double const *const *parameters,
                                              double *residuals,
                                              double **jacobians) const
{
  {
    Eigen::Vector3d twc(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond qwc(parameters[0][6], parameters[0][3], parameters[0][4],
                           parameters[0][5]);
    Eigen::Vector3d vanishing_direction(parameters[1][0], parameters[1][1], parameters[1][2]);
    // std::cout << "para_line_4:" << vanishing_direction << std::endl;
    (*residuals) = residual_from_riemannian_rotation(qwc.toRotationMatrix(), vanishing_direction, obs_i);
    // std::cout << "residual:" << residual << std::endl;
    sqrt_info.setIdentity();
  }
  if (jacobians)
  {
    const double eps = 1e-6;
    Eigen::Matrix<double, 1, jacobian_pose_lines_parameters> num_jacobian =
        Eigen::Matrix<double, 1, jacobian_pose_lines_parameters>::Zero();
    for (int k = 0; k < jacobian_pose_lines_parameters; k++)
    {
      Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
      Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3],
                            parameters[0][4], parameters[0][5]);

      Eigen::Vector3d vanishing_dire(parameters[1][0], parameters[1][1], parameters[1][2]);
      int a = k / 3, b = k % 3;
      Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

      if (a == 1 && jacobians[0])
        Qi = Qi * UtiliLine::deltaQ(delta);
      else if (k == 6 || k == 7)
      {
        Eigen::Vector2d delta_riemannian;

        if (b == 0)
          delta_riemannian(0) += eps;
        if (b == 1)
          delta_riemannian(1) += eps;
        // Eigen::Matrix<double, 2, 1> delta_riemannian = Eigen::Matrix<double, 2, 1>(k == 6, k == 7) * eps;
        // line orth的前三个元素
        if (jacobians[1])
          vanishing_dire = UtiliLine::delta_vanishing_direc(vanishing_dire, delta_riemannian);
      }

      double tmp_residual = residual_from_riemannian_rotation(Qi.toRotationMatrix(), vanishing_dire, obs_i);

      double residual = (*residuals);
      num_jacobian(0, k) = (tmp_residual - residual) / eps;
    }

    if (jacobians[0])
    {
      Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose(
          jacobians[0]);
      jacobian_pose.leftCols<3>().setZero();
      jacobian_pose.block(0, 3, 1, 3) = num_jacobian.block(0, 3, 1, 3); // leftCols<6>();
      jacobian_pose.rightCols<1>().setZero();
    }
    if (jacobians[1])
    {
      Eigen::Map<Eigen::Matrix<double, 1, global_size_vec, Eigen::RowMajor>>
          jacobian_lineOrth_shared(jacobians[1]);
      jacobian_lineOrth_shared.leftCols<2>() = num_jacobian.block(0, 6, 1, 2);
      jacobian_lineOrth_shared.rightCols<unique_size_vec>().setZero();
    }
  }
  return true;
}

// Eigen::Matrix2d RiemanianlinesetRotationFactor2::sqrt_info;
// double RiemanianlinesetRotationFactor2::sum_t;
// RiemanianlinesetRotationFactor2::RiemanianlinesetRotationFactor2(const Eigen::Vector4d &_pts_i) : obs_i(_pts_i) {};

// bool RiemanianlinesetRotationFactor2::Evaluate(double const *const *parameters,
//                                                double *residuals,
//                                                double **jacobians) const
// {
//   {
//     Eigen::Vector3d twc(parameters[0][0], parameters[0][1], parameters[0][2]);
//     Eigen::Quaterniond qwc(parameters[0][6], parameters[0][3], parameters[0][4],
//                            parameters[0][5]);
//     Eigen::Vector3d vanishing_direction(parameters[1][0], parameters[1][1], parameters[1][2]);
//     // std::cout<<"para_line_4:"<< para_line_4<<std::endl;
//     (*residuals) = residual_from_riemannian_rotation(qwc.toRotationMatrix(), vanishing_direction, obs_i);
//     // std::cout<<"residual:"<<residual<<std::endl;
//     sqrt_info.setIdentity();
//   }
//   if (jacobians)
//   {
//     const double eps = 1e-6;
//     Eigen::Matrix<double, 1, jacobian_pose_lines_parameters_16> num_jacobian =
//         Eigen::Matrix<double, 1, jacobian_pose_lines_parameters_16>::Zero();
//     for (int k = 0; k < jacobian_pose_lines_parameters_16; k++)
//     {
//       Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
//       Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3],
//                             parameters[0][4], parameters[0][5]);

//       Eigen::Vector3d vanishing_dire(parameters[1][0], parameters[1][1], parameters[1][2]);
//       int a = k / 3, b = k % 3;
//       Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

//       if (a == 1 && jacobians[0])
//         Qi = Qi * UtiliLine::deltaQ(delta);
//       else if (k == 6 || k == 7)
//       {
//         Eigen::Matrix<double, 2, 1> delta_riemannian = Eigen::Matrix<double, 2, 1>(k == 6, k == 7) * eps;

//         // line orth的前三个元素
//         if (jacobians[1])
//           vanishing_dire = UtiliLine::delta_vanishing_direc(vanishing_dire, delta_riemannian);
//       }

//       double tmp_residual = residual_from_riemannian_rotation(Qi.toRotationMatrix(), vanishing_dire, obs_i);

//       double residual = (*residuals);
//       num_jacobian(0, k) = (tmp_residual - residual) / eps;
//     }

//     if (jacobians[0])
//     {
//       Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose(
//           jacobians[0]);
//       jacobian_pose.leftCols<3>().setZero();
//       jacobian_pose.block(0, 3, 1, 3) = num_jacobian.block(0, 3, 1, 3); // leftCols<6>();
//       jacobian_pose.rightCols<1>().setZero();
//     }
//     if (jacobians[1])
//     {
//       Eigen::Map<Eigen::Matrix<double, 1, global_size_vec_16, Eigen::RowMajor>>
//           jacobian_lineOrth_shared(jacobians[1]);
//       jacobian_lineOrth_shared.leftCols<2>() = num_jacobian.block(0, 6, 1, 2);
//       jacobian_lineOrth_shared.rightCols<unique_size_vec_16>().setZero();
//     }
//   }
//   return true;
// }

// Eigen::Matrix2d RiemanianlinesetRotationFactor3::sqrt_info;
// double RiemanianlinesetRotationFactor3::sum_t;
// RiemanianlinesetRotationFactor3::RiemanianlinesetRotationFactor3(const Eigen::Vector4d &_pts_i) : obs_i(_pts_i) {};

// bool RiemanianlinesetRotationFactor3::Evaluate(double const *const *parameters,
//                                                double *residuals,
//                                                double **jacobians) const
// {
//   {
//     Eigen::Vector3d twc(parameters[0][0], parameters[0][1], parameters[0][2]);
//     Eigen::Quaterniond qwc(parameters[0][6], parameters[0][3], parameters[0][4],
//                            parameters[0][5]);
//     Eigen::Vector3d vanishing_direction(parameters[1][0], parameters[1][1], parameters[1][2]);
//     // std::cout<<"para_line_4:"<< para_line_4<<std::endl;
//     (*residuals) = residual_from_riemannian_rotation(qwc.toRotationMatrix(), vanishing_direction, obs_i);
//     // std::cout<<"residual:"<<residual<<std::endl;
//     sqrt_info.setIdentity();
//   }
//   if (jacobians)
//   {
//     const double eps = 1e-6;
//     Eigen::Matrix<double, 1, jacobian_pose_lines_parameters_n3> num_jacobian =
//         Eigen::Matrix<double, 1, jacobian_pose_lines_parameters_n3>::Zero();
//     for (int k = 0; k < jacobian_pose_lines_parameters_n3; k++)
//     {
//       Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
//       Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3],
//                             parameters[0][4], parameters[0][5]);

//       Eigen::Vector3d vanishing_dire(parameters[1][0], parameters[1][1], parameters[1][2]);
//       int a = k / 3, b = k % 3;
//       Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

//       if (a == 1 && jacobians[0])
//         Qi = Qi * UtiliLine::deltaQ(delta);
//       else if (k == 6 || k == 7)
//       {
//         Eigen::Matrix<double, 2, 1> delta_riemannian = Eigen::Matrix<double, 2, 1>(k == 6, k == 7) * eps;

//         // line orth的前三个元素
//         if (jacobians[1])
//           vanishing_dire = UtiliLine::delta_vanishing_direc(vanishing_dire, delta_riemannian);
//       }

//       double tmp_residual = residual_from_riemannian_rotation(Qi.toRotationMatrix(), vanishing_dire, obs_i);

//       double residual = (*residuals);
//       num_jacobian(0, k) = (tmp_residual - residual) / eps;
//     }

//     if (jacobians[0])
//     {
//       Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose(
//           jacobians[0]);
//       jacobian_pose.leftCols<3>().setZero();
//       jacobian_pose.block(0, 3, 1, 3) = num_jacobian.block(0, 3, 1, 3); // leftCols<6>();
//       jacobian_pose.rightCols<1>().setZero();
//     }
//     if (jacobians[1])
//     {
//       Eigen::Map<Eigen::Matrix<double, 1, global_size_vec_n3, Eigen::RowMajor>>
//           jacobian_lineOrth_shared(jacobians[1]);
//       jacobian_lineOrth_shared.leftCols<2>() = num_jacobian.block(0, 6, 1, 2);
//       jacobian_lineOrth_shared.rightCols<unique_size_vec_n3>().setZero();
//     }
//   }
//   return true;
// }
