/***
 * @Author: yanyan-li yanyan.li.camp@gmail.com
 * @Date: 2023-11-21 12:31:15
 * @LastEditTime: 2023-11-21 14:39:52
 * @LastEditors: yanyan-li yanyan.li.camp@gmail.com
 * @Description:
 * @FilePath: /LineRepresentation/src/optimizer/factor/RiemanlinesProjectionFactor.hpp
 */

#pragma once

#include "../../utils/UtilParameterss.hpp"
#include <Eigen/Dense>
#include <ceres/ceres.h>

typedef Eigen::Matrix<double, 6, 1> Vector6d;

/**
 * @brief for any size of line set
 *
 */
class RiemanianlinesetProjectionFactorX : public ceres::CostFunction
{
public:
    RiemanianlinesetProjectionFactorX(const Eigen::Vector4d &_pts_i, int _position_i, int num_lines);

    virtual bool Evaluate(double const *const *parameters,
                          double *residuals,
                          double **jacobians) const override;
    void check(double **parameters);

private:
    Eigen::Vector4d obs_i;
    int position;  // Line index in the group
    int num_lines; // Total number of lines in group
    static Eigen::Matrix2d sqrt_info;
    static double sum_t;
    Vector6d PLK0;
    // Eigen::Matrix<double, 3, 2> TangentBasisVectors;
    Eigen::Vector3d TngentBaseVector;
};
