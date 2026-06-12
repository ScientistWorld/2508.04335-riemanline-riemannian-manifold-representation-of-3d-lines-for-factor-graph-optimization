/*
 * @Author: yanyan-li yanyan.li.camp@gmail.com
 * @Date: 2023-10-21 09:13:58
 * @LastEditTime: 2023-10-21 15:22:35
 * @LastEditors: yanyan-li yanyan.li.camp@gmail.com
 * @Description:
 * @FilePath: /LineRepresentation/examples/test.cc
 */

#include "src/utils/UtilTransformer.hpp"
#include <eigen3/Eigen/Dense>
#include <iostream>

void fcn_rotation_from_two_vectors(Eigen::Vector3d &v1,
                                   Eigen::Vector3d &v2,
                                   Eigen::Matrix3d &rotation);
void fcn_get_skew(Eigen::Vector3d &w, Eigen::Matrix3d &W);
int main()
{
    Eigen::Vector3d a_x, a_y, b_x;
    a_x << 1, 2, 3;
    a_x = a_x / a_x.norm();
    a_y << 0, 4, 5;
    a_y = a_x.cross(a_y);
    a_y = a_y / a_y.norm();
    b_x << 2, 3, 4;
    b_x = b_x / b_x.norm();

    // -------->
    // tangent space of a_x
    Eigen::Matrix<double, 3, 2> TangentBasisVectors = UtiliLine::TangentBasis(a_x);
    // step
    double alpha = 0.002;
    double beta = 0.0016;
    double gamma = 0.001;
    Eigen::Vector3d _manifold_direc_move = alpha * TangentBasisVectors.col(0) + beta * TangentBasisVectors.col(1);
    std::cout << "tangent:" << _manifold_direc_move << std::endl;
    Eigen::Vector3d _direct = a_x.tail(3) * cos(_manifold_direc_move.norm()) + _manifold_direc_move / (_manifold_direc_move.norm()) * sin(_manifold_direc_move.norm());

    std::cout << "d:" << _direct << ", direct: " << _direct.norm() << ", a_x" << a_x << std::endl;
    Eigen::Matrix3d rot(Eigen::Matrix3d::Identity());

    // rot from a_x to b_x
    fcn_rotation_from_two_vectors(a_x, _direct, rot);
    std::cout << "rot motion:" << _direct << "," << rot * a_x << std::endl;
    std::cout << "rot a_y to b_y:" << a_y << "," << rot * a_y << std::endl;
    double circle_check = (rot * a_y).transpose() * _direct;
    std::cout << "circle_check:" << circle_check << std::endl;

    Eigen::Vector3d tangent_vector = _direct.cross(rot * a_y);
    tangent_vector = tangent_vector / tangent_vector.norm();
    Eigen::Vector3d _manifold_norm_move = tangent_vector * gamma;
    Eigen::Vector3d _normal = (rot * a_y) * cos(_manifold_norm_move.norm()) + _manifold_norm_move / (_manifold_norm_move.norm()) * sin(_manifold_norm_move.norm());

    std::cout << "circle_check 2:" << _normal.transpose() * _direct << std::endl;
    return 0;
}

void fcn_rotation_from_two_vectors(Eigen::Vector3d &v1,
                                   Eigen::Vector3d &v2,
                                   Eigen::Matrix3d &rotation)
{
    Eigen::Vector3d w = v1.cross(v2);
    // std::cout << v1 << "," << v2 << std::endl;
    // std::cout << "w:" << w << ", norm:" << w.norm();
    w = w / w.norm();
    Eigen::Matrix3d W;
    fcn_get_skew(w, W);
    std::cout << "w:" << w << ",W:" << W;

    double cos_tht = (v1.transpose() * v2);
    double len = (v1.norm() * v2.norm());
    cos_tht = cos_tht / len;
    double tht = acos(cos_tht);
    std::cout << "angle:" << cos_tht << "," << tht << std::endl;

    rotation = Eigen::Matrix3d::Identity() + W * sin(tht) + W * W * (1 - cos(tht));
}

void fcn_get_skew(Eigen::Vector3d &w, Eigen::Matrix3d &w_skew)
{
    w_skew << 0, -w(2), w(1),
        w(2), 0, -w(0),
        -w(1), w(0), 0;
}