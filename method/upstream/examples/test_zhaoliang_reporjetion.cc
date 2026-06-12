#include <iostream>
#include <Eigen/Core>
#include "src/utils/UtilTransformer.hpp"

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

  std::cout << "norm_0: " << norm_0.transpose() << std::endl;
  std::cout << "norm_1: " << norm_1.transpose() << std::endl;

  Eigen::Vector3d norm_i = (twc1 - twci).dot(norm_1) * norm_0 - (twc0 - twci).dot(norm_0) * norm_1;

  norm_i = Rwci.transpose() * norm_i;


//   norm_i = norm_i / norm_i.norm();

  std::cout << "norm_i: " << norm_i.transpose() << std::endl;

  double l_norm = norm_i(0) * norm_i(0) + norm_i(1) * norm_i(1);
  double l_sqrtnorm = sqrt(l_norm);

  double e1 = obs_i(0) * norm_i(0) + obs_i(1) * norm_i(1) + norm_i(2);
  double e2 = obs_i(2) * norm_i(0) + obs_i(3) * norm_i(1) + norm_i(2);

  Eigen::Vector2d residual;
  residual(0) = e1 / l_sqrtnorm;
  residual(1) = e2 / l_sqrtnorm;

  return residual;
}


int main() {
    // 1. generate 3D line and 3 poses, and related 2D line observations
    Eigen::Vector3d line_3d_0(0, 0, 2);
    Eigen::Vector3d line_3d_1(2, 2, 2);

    std::vector<Eigen::Vector3d> pose_3d;  //t_wi
    Eigen::Vector3d pose_0(1, 0, 0);
    Eigen::Vector3d pose_1(1, 1, 0);
    Eigen::Vector3d pose_2(2, 1, 0);
    pose_3d.push_back(pose_0);
    pose_3d.push_back(pose_1);
    pose_3d.push_back(pose_2);
    std::vector<Eigen::Matrix3d> Rs_wi;
    Rs_wi.push_back(Eigen::AngleAxisd(M_PI / 8, Eigen::Vector3d(0, 0, 1)).toRotationMatrix());
    Rs_wi.push_back(Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d(0, 0, 1)).toRotationMatrix());
    Rs_wi.push_back(Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix());
    // Eigen::Matrix3d R_wi = Eigen::Matrix3d::Identity();

    std::vector<Eigen::Vector3d> line_2d_0;
    std::vector<Eigen::Vector3d> line_2d_1;

    for (int i = 0; i < pose_3d.size(); i++) {
        Eigen::Vector3d pose = pose_3d[i];

        Eigen::Matrix3d R_iw = Rs_wi[i].transpose();
        Eigen::Vector3d t_iw = -R_iw * pose;

        Eigen::Vector3d line_3d_0_ = R_iw * line_3d_0 + t_iw;
        Eigen::Vector3d line_3d_1_ = R_iw * line_3d_1 + t_iw;

        Eigen::Vector3d line_3d_0_norm = line_3d_0_ / line_3d_0_(2);
        Eigen::Vector3d line_3d_1_norm = line_3d_1_ / line_3d_1_(2);

        line_2d_0.push_back(line_3d_0_norm);
        line_2d_1.push_back(line_3d_1_norm);
    }


    // 2. compute the reprojection error

    Eigen::Matrix3d Rwci = Rs_wi[0];
    Eigen::Vector3d twci = pose_3d[0];
    Eigen::Matrix3d Rwc0 = Rs_wi[1];
    Eigen::Vector3d twc0 = pose_3d[1];
    Eigen::Matrix3d Rwc1 = Rs_wi[2];
    Eigen::Vector3d twc1 = pose_3d[2];

    Eigen::Vector4d line_zhaoliang;
    Eigen::Vector3d norm_0 = line_2d_0[1].cross(line_2d_1[1]);
    norm_0 = norm_0 / norm_0.norm();
    norm_0 = Rwc0 * norm_0;
    std::cout << "norm_0: " << norm_0.transpose() << std::endl;
    line_zhaoliang.head(2) = UtiliLine::Normvector2Angle(norm_0);
    Eigen::Vector3d norm_1 = line_2d_0[2].cross(line_2d_1[2]);
    norm_1 = norm_1 / norm_1.norm();
    norm_1 = Rwc1 * norm_1;
    std::cout << "norm_1: " << norm_1.transpose() << std::endl;
    line_zhaoliang.tail(2) = UtiliLine::Normvector2Angle(norm_1);
    // std::cout << "line_zhaoliang: " << line_zhaoliang.transpose() << std::endl;

    Eigen::Vector3d norm_i = line_2d_0[0].cross(line_2d_1[0]);
    norm_i = norm_i / norm_i.norm();
    std::cout << "norm_i: " << norm_i.transpose() << std::endl;


    std::cout << "line_zhaoliang: " << line_zhaoliang.transpose() << std::endl;

    Eigen::Vector4d obs_i;
    obs_i.head(2) = line_2d_0[0].head(2);
    obs_i.tail(2) = line_2d_1[0].head(2);

    Eigen::Vector2d error = residual_from_zhaoliang(Rwci, twci, Rwc0, twc0, Rwc1, twc1, line_zhaoliang, obs_i);

    std::cout << "error: " << error.transpose() << std::endl;


    return 0;
}
