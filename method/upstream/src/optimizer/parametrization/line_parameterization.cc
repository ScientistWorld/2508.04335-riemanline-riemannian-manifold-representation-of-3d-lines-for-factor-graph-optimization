#include "line_parameterization.hpp"
#include "src/utils/UtilTransformer.hpp"

bool LineOrthParameterization::Plus(const double *x,
                                    const double *delta,
                                    double *x_plus_delta) const
{
    // ref: 2001, Adrien Bartol,Peter Sturm ,Structure-From-Motion Using Lines:
    // Representation, Triangulation and Bundle Adjustment

    // theta --> U,  phi --> W
    Eigen::Map<const Eigen::Vector3d> theta(x);
    double phi = *(x + 3);
    // Vector3d theta = orth.head(3);
    // double phi = orth[3];
    double s1 = sin(theta[0]); // x, varphi
    double c1 = cos(theta[0]);
    double s2 = sin(theta[1]); // y theta
    double c2 = cos(theta[1]);
    double s3 = sin(theta[2]); // z phi
    double c3 = cos(theta[2]);
    // from eular to rotation
    // cos θ*cos φ,  sin ψ sin θ cos φ − cos ψ sin φ cos ψ sin θ cos φ + sin ψ sin φ
    // cos θ sin φ sin ψ sin θ sin φ + cos ψ cos φ cos ψ sin θ sin φ − sin ψ cos φ
    // − sin θ sin ψ cos θ cos ψ cos θ
    Eigen::Matrix3d R;
    R << c2 * c3, s1 * s2 * c3 - c1 * s3, c1 * s2 * c3 + s1 * s3, c2 * s3,
        s1 * s2 * s3 + c1 * c3, c1 * s2 * s3 - s1 * c3, -s2, s1 * c2, c1 * c2;
    double w1 = cos(phi);
    double w2 = sin(phi);

    // update
    Eigen::Map<const Eigen::Vector3d> _delta_theta(delta);
    double _delta_phi = *(delta + 3);
    Eigen::Matrix3d Rz;
    Rz << cos(_delta_theta(2)), -sin(_delta_theta(2)), 0, sin(_delta_theta(2)),
        cos(_delta_theta(2)), 0, 0, 0, 1;

    Eigen::Matrix3d Ry;
    Ry << cos(_delta_theta(1)), 0., sin(_delta_theta(1)), 0., 1., 0.,
        -sin(_delta_theta(1)), 0., cos(_delta_theta(1));

    Eigen::Matrix3d Rx;
    Rx << 1., 0., 0., 0., cos(_delta_theta(0)), -sin(_delta_theta(0)), 0.,
        sin(_delta_theta(0)), cos(_delta_theta(0));
    R = R * Rx * Ry * Rz;

    Eigen::Matrix2d W;
    W << w1, -w2, w2, w1;
    Eigen::Matrix2d delta_W;
    delta_W << cos(_delta_phi), -sin(_delta_phi), sin(_delta_phi),
        cos(_delta_phi);
    W = W * delta_W;

    // U' -- > theta'. W' --> phi'
    Eigen::Map<Eigen::Vector3d> theta_pluse(
        x_plus_delta); // double 指针 转为eigen数组
    double *phi_plus(x_plus_delta + 3);

    Eigen::Vector3d u1 = R.col(0);
    Eigen::Vector3d u2 = R.col(1);
    Eigen::Vector3d u3 = R.col(2);
    //   theta_pluse[0] = atan2(u2(2), u3(2));
    //   theta_pluse[1] = asin(-u1(2));
    //   theta_pluse[2] = atan2(u1(1), u1(0));

    double sy = sqrt(u1(0) * u1(0) + u1(1) * u1(1));
    // float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
    // std::cout<<"sy:"<<sy<<std::endl;
    assert(sy > 1e-6);
    // todo:: use SO3
    //   // orth[0] = atan2(u2(2), u3(2));
    //   // orth[1] = asin(-u1(2));
    //   // orth[2] = atan2(u1(1), u1(0));
    theta_pluse[0] = atan2(u2(2), u3(2));
    theta_pluse[1] = atan2(-u1(2), sy); // asin(-u1(2));
    theta_pluse[2] = atan2(u1(1), u1(0));

    *phi_plus = asin(W(1, 0));
    return true;
}

bool LineRiemanParameterization::Plus(const double *x,
                                      const double *delta,
                                      double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> plk(x);

    Eigen::Matrix<double, 3, 2> TangentBasisVectors = UtiliLine::TangentBasis(plk.tail(3));
    // Eigen::Vector3d TangentVector = UtiliLine::TangentVector(plk.tail(3), plk.head(3)); // for normal
    // Eigen::Vector4d LineOrth = UtiliLine::plk_to_orth(plk);
    // double phi = LineOrth[3];
    // double w1 = cos(phi);
    // double w2 = sin(phi);
    double d = plk.head(3).norm();

    //---> update on manifold
    // Eigen::Map<const Eigen::Vector3d> _delta_theta(delta);
    Eigen::Map<const Eigen::Vector4d> delta_vec(delta);
    double _direct_step0 = delta_vec[0];
    double _direct_step1 = delta_vec[1];
    double _normal_step = delta_vec[2];
    double _scale_step = delta_vec[3];

    // double _direct_step0 = *(delta);
    // double _direct_step1 = *(delta + 1);
    // double _normal_step = *(delta + 2);
    Eigen::Vector3d _manifold_direc_move = _direct_step0 * TangentBasisVectors.col(0) + _direct_step1 * TangentBasisVectors.col(1);
    double move_norm = _manifold_direc_move.norm();
    // from TS^2 to S2
    Eigen::Vector3d _direct;
    if (move_norm < 1e-8)
        _direct = plk.tail(3);
    else
    {
        _direct = plk.tail(3) * cos(move_norm) + _manifold_direc_move / move_norm * sin(move_norm);
        // lying on the sphere
        _direct.normalize();
    }

    // from TS^2 to S2
    // Eigen::Vector3d _direct = plk.tail(3) * cos(_manifold_direc_move.norm()) + _manifold_direc_move / (_manifold_direc_move.norm()) * sin(_manifold_direc_move.norm());

    // align normal (TODO: use rotation)
    Eigen::Vector3d _normal = (plk.head(3).normalized() - _direct * (_direct.transpose() * plk.head(3).normalized())).normalized();
    Eigen::Vector3d TangentVector = UtiliLine::TangentVector(_direct, _normal); // for normal

    Eigen::Vector3d _manifold_normal_move = _normal_step * TangentVector;
    // from TS to S
    _normal = _normal * cos(_manifold_normal_move.norm()) + _manifold_normal_move / (_manifold_normal_move.norm()) * sin(_manifold_normal_move.norm());
    // SO(2) update
    d += *(delta + 3);
    // Eigen::Matrix2d W;
    // W << w1, -w2, w2, w1;
    // Eigen::Matrix2d delta_W;
    // delta_W << cos(_delta_phi), -sin(_delta_phi), sin(_delta_phi),
    //     cos(_delta_phi);
    // W = W * delta_W;

    // //---> reback
    // phi = asin(W(1, 0));
    // w1 = cos(phi);
    // _normal = w1 * _normal;
    _normal = d * _normal;
    // make sure the direction aligned with the origin
    if (plk.head(3).dot(_normal) < 0)
        _normal = -_normal;

    Eigen::Map<Eigen::Matrix<double, 6, 1>>
        theta_pluse(x_plus_delta);
    theta_pluse[3] = _direct.x();
    theta_pluse[4] = _direct.y();
    theta_pluse[5] = _direct.z();
    theta_pluse[0] = _normal.x();
    theta_pluse[1] = _normal.y();
    theta_pluse[2] = _normal.z();
    return true;
}

bool LineParaRiemanLineParameterization::Plus(const double *x,
                                              const double *delta,
                                              double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> d(x);
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> delta_all(delta, LocalSize());

    // Step 1: update d using tangent basis
    Eigen::Matrix<double, 3, 2> d_basis = UtiliLine::TangentBasis(d);

    // std::cout << "Plus d_basis:" << d_basis << std::endl;

    Eigen::Vector2d delta_d = delta_all.segment<2>(0);
    Eigen::Vector3d d_updated = (d + d_basis * delta_d).normalized();
    Eigen::Map<Eigen::Vector3d> d_out(x_plus_delta);
    d_out = d_updated;

    // Step 2: update each normal (in plane orthogonal to d)
    for (int i = 0; i < num_lines_; ++i)
    {
        Eigen::Map<const Eigen::Vector3d> n_i(x + 3 + 3 * i);
        Eigen::Matrix<double, 3, 2> n_basis = UtiliLine::TangentBasis(d); // tangent space to d

        Eigen::Vector2d delta_n = delta_all.segment<2>(2 + 2 * i);
        Eigen::Vector3d delta_vec = n_basis * delta_n;

        Eigen::Map<Eigen::Vector3d> n_out_i(x_plus_delta + 3 + 3 * i);
        n_out_i = n_i + delta_vec;
    }

    return true;
    // Shared direction
    // Eigen::Map<const Eigen::Vector3d> direction(x);
    // Eigen::Matrix<double, 3, 2> basis = UtiliLine::TangentBasis(direction.normalized());

    // Eigen::Vector2d dir_delta(delta[0], delta[1]);
    // double theta = dir_delta.norm();

    // Eigen::Vector3d updated_dir = direction.normalized();
    // if (theta > 1e-10)
    // {
    //     Eigen::Vector3d move = basis * dir_delta;
    //     updated_dir = std::cos(theta) * updated_dir + std::sin(theta) * move.normalized();
    //     updated_dir.normalize();
    // }

    // // 写入方向
    // Eigen::Map<Eigen::Vector3d> dir_out(x_plus_delta);
    // dir_out = updated_dir;

    // for (int i = 0; i < num_lines_; ++i)
    // {
    //     Eigen::Vector3d n(x[3 + 3 * i], x[3 + 3 * i + 1], x[3 + 3 * i + 2]);
    //     double norm_n = n.norm();
    //     if (norm_n < 1e-8)
    //         continue;

    //     Eigen::Vector3d n_proj = n - updated_dir * updated_dir.dot(n);
    //     n_proj.normalize();

    //     double step_rot = delta[2 + 2 * i];
    //     double step_scale = delta[2 + 2 * i + 1];

    //     Eigen::Vector3d rot_axis = UtiliLine::TangentVector(updated_dir, n_proj);
    //     double theta_n = std::abs(step_rot);

    //     if (theta_n > 1e-8)
    //     {
    //         rot_axis.normalize();
    //         n_proj = std::cos(theta_n) * n_proj + std::sin(theta_n) * rot_axis;
    //     }

    //     Eigen::Vector3d n_new = n_proj.normalized() * (norm_n + step_scale);
    //     if (n_new.dot(updated_dir) < 0)
    //         n_new = -n_new;

    //     Eigen::Map<Eigen::Vector3d> n_out(x_plus_delta + 3 + 3 * i);
    //     n_out = n_new;
    // }

    // return true;

    ///----------------------------------------------->3
    // const int global_size = 3 * num_lines_ + 3; // or whatever global_size_vec resolves to
    // Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> theta_plus(x_plus_delta, global_size);

    // Eigen::Vector3d direction(x[0], x[1], x[2]);
    // direction.normalize();

    // // Tangent basis for Riemannian update
    // Eigen::Matrix<double, 3, 2> tangent_basis = UtiliLine::TangentBasis(direction);
    // // std::cout << "tangent_basis:" << direction << "," << tangent_basis.block<3, 1>(0, 0) << ", " << tangent_basis.block<3, 1>(0, 1) << std::endl;

    // // Apply delta on the sphere (TS^2 manifold)
    // Eigen::Vector2d direction_delta(delta[0], delta[1]);
    // double delta_norm = direction_delta.norm();

    // Eigen::Vector3d updated_direction = direction;
    // if (delta_norm > 1e-8)
    // {
    //     Eigen::Vector3d move = tangent_basis * direction_delta;
    //     updated_direction = direction * std::cos(delta_norm) + (move / delta_norm) * std::sin(delta_norm);
    //     updated_direction.normalize();
    // }

    // // Write back updated direction
    // theta_plus.segment<3>(0) = updated_direction;

    // // Step 2: Handle all remaining normal vectors
    // for (int i = 1; i < num_lines_ + 1; ++i)
    // {
    //     Eigen::Vector3d normal(x[3 * i + 0], x[3 * i + 1], x[3 * i + 2]);

    //     double distance = normal.norm();
    //     // if (distance < 1e-6)
    //     //     distance = 1e-6;

    //     normal.normalize();

    //     // Project current normal to updated direction's tangent space
    //     Eigen::Vector3d proj = normal - updated_direction * (updated_direction.dot(normal));
    //     proj.normalize(); // ensure orthogonality

    //     // Retrieve delta values for normal update
    //     double step_along_tangent = delta[2 * i];
    //     double step_scale = delta[2 * i + 1]; // scaling the magnitude

    //     // Use the tangent vector (rotation axis) to rotate the normal
    //     Eigen::Vector3d tangent_vec = UtiliLine::TangentVector(updated_direction, proj);
    //     double rot_norm = std::abs(step_along_tangent);

    //     if (rot_norm > 1e-8)
    //     {
    //         Eigen::Vector3d rot = tangent_vec.normalized() * step_along_tangent;
    //         normal = proj * std::cos(rot_norm) + (rot / rot_norm) * std::sin(rot_norm);
    //     }
    //     else
    //     {
    //         normal = proj;
    //     }

    //     normal.normalize();
    //     normal *= (distance + step_scale); // apply scale change
    //     if (updated_direction.dot(normal) < 0)
    //         normal = -normal;

    //     theta_plus.segment<3>(3 * i) = normal;
    // }

    // return true;
}

bool ParaRiemanLineParameterization::Plus(const double *x,
                                          const double *delta,
                                          double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> d(x);
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> delta_all(delta, LocalSize());

    // Step 1: update d using tangent basis
    Eigen::Matrix<double, 3, 2> d_basis = UtiliLine::TangentBasis(d);

    // std::cout << "Plus d_basis:" << d_basis << std::endl;
    Eigen::Vector2d delta_d = delta_all.segment<2>(0);
    // double norm_delta_d = delta_d.norm();

    // Eigen::Vector3d _manifold_direc_move = _direct_step0 * d_basis.col(0) + _direct_step1 * d_basis.col(1);

    Eigen::Vector3d _manifold_direc_move = (d + d_basis * delta_d).normalized();

    double move_norm = _manifold_direc_move.norm();
    // from TS^2 to S2
    Eigen::Vector3d _direct;
    if (move_norm < 1e-8)
        _direct = d;
    else
    {
        _direct = d * cos(move_norm) + _manifold_direc_move / move_norm * sin(move_norm);
        // lying on the sphere
        _direct.normalize();
    }

    // std::cout << "---->original vs updated:" << d << "," << _direct
    //           << ";" << delta_d << "," << d_basis << std::endl;
    Eigen::Map<Eigen::Vector3d> d_out(x_plus_delta);
    d_out = _direct;

    // Step 2: update each normal (in plane orthogonal to d)
    for (int i = 0; i < num_lines_; ++i)
    {
        // Eigen::Map<const Eigen::Vector3d> n_i(x + 3 + 3 * i);
        // Eigen::Matrix<double, 3, 2> n_basis = UtiliLine::TangentBasis(d); // tangent space to d

        Eigen::Vector3d n_i(x[3 + 3 * i], x[3 + 3 * i + 1], x[3 + 3 * i + 2]);
        double norm_n = n_i.norm();
        Eigen::Vector3d n_init = n_i.normalized();
        if (norm_n < 1e-8)
            continue;

        Eigen::Vector3d n_proj = n_init - _direct * _direct.dot(n_init);
        n_proj.normalize();

        double step_rot = delta[2 + 2 * i];
        double step_scale = delta[2 + 2 * i + 1];

        Eigen::Vector3d rot_axis = _direct.cross(n_proj).normalized();
        Eigen::Vector3d n_rotated;
        if (abs(step_rot) > 1e-8)
            n_rotated = n_proj * cos(step_rot) + rot_axis * sin(step_rot);
        else
            n_rotated = n_proj;

        // Eigen::Vector3d rot_axis = UtiliLine::TangentVector(d_updated, n_proj);
        // Eigen::Vector3d _manifold_normal_move = step_rot * rot_axis;
        // n_proj = n_proj * cos(_manifold_normal_move.norm()) + _manifold_normal_move / (_manifold_normal_move.norm()) * sin(_manifold_normal_move.norm());
        // // SO(2) update
        norm_n += step_scale;
        n_rotated = norm_n * n_rotated;
        Eigen::Map<Eigen::Vector3d> n_out_i(x_plus_delta + 3 + 3 * i);
        n_out_i = n_rotated;
    }

    return true;
}

bool ParaRiemanLineParameterization::ComputeJacobian(const double *x,
                                                     double *jacobian) const
{
    const int global_size = GlobalSize(); // e.g., 3 + 3*num_lines_
    const int local_size = LocalSize();   // e.g., 2 + 2*num_lines_
    const double eps = 1e-6;

    Eigen::Map<const Eigen::VectorXd> x_vec(x, global_size);
    Eigen::MatrixXd J(global_size, local_size);

    for (int i = 0; i < local_size; ++i)
    {
        Eigen::VectorXd delta = Eigen::VectorXd::Zero(local_size);
        delta(i) = eps;

        std::vector<double> x_plus_delta(global_size);
        Plus(x, delta.data(), x_plus_delta.data());

        for (int j = 0; j < global_size; ++j)
        {
            J(j, i) = (x_plus_delta[j] - x[j]) / eps;
        }
    }

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> j_out(jacobian, global_size, local_size);
    j_out = J;

    // Eigen::Map<const Eigen::Vector3d> d(x);

    // Eigen::Matrix<double, 3, 2> d_basis = UtiliLine::TangentBasis(d);
    // Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
    //     J(jacobian, GlobalSize(), LocalSize());
    // J.setZero();

    // // J.topRows(LocalSize()).setIdentity();

    // if (true)
    // {
    //     // Part 1: ∂d / ∂delta_d
    //     J.block<3, 2>(0, 0) = d_basis;

    //     // Part 2: ∂n_i / ∂delta_rot_i and ∂n_i / ∂delta_scale_i
    //     // Eigen::Vector3d d_unit = d.normalized();

    //     // for (int i = 0; i < num_lines_; ++i)
    //     // {
    //     //     const Eigen::Map<const Eigen::Vector3d> n_i(x + 3 + 3 * i);
    //     //     double norm_n = n_i.norm();
    //     //     if (norm_n < 1e-8)
    //     //         continue;

    //     //     Eigen::Vector3d n_init = n_i.normalized();
    //     //     Eigen::Vector3d n_proj = n_init - d_unit * d_unit.dot(n_init);
    //     //     n_proj.normalize();

    //     //     Eigen::Vector3d rot_axis = UtiliLine::TangentVector(d_unit, n_proj);
    //     //     // double rot_axis_norm = rot_axis.norm();
    //     //     // if (rot_axis_norm < 1e-8)
    //     //     //     continue;

    //     //     rot_axis.normalize();

    //     //     // Derivative of rotated vector using small-angle approx: R(n_proj) ≈ n_proj + θ × n_proj
    //     //     Eigen::Vector3d dn_dtheta = rot_axis.cross(n_proj);

    //     //     // Final projected and scaled normal: norm_n * R(n_proj)
    //     //     Eigen::Matrix<double, 3, 2> J_n;
    //     //     J_n.col(0) = norm_n * dn_dtheta; // ∂n_proj / ∂rot
    //     //     J_n.col(1) = n_proj;             // ∂n_proj / ∂scale

    //     //     J.block<3, 2>(3 + 3 * i, 2 + 2 * i) = J_n;
    //     // }

    //     // Jacobians for each normal
    //     for (int i = 0; i < num_lines_; ++i)
    //     {
    //         Eigen::Vector3d n_i(x + 3 + 3 * i);
    //         double norm_n = n_i.norm();
    //         if (norm_n < 1e-8)
    //             continue;

    //         Eigen::Vector3d n_init = n_i.normalized();
    //         Eigen::Vector3d n_proj = n_init - d * d.dot(n_init);
    //         n_proj.normalize();

    //         double step_rot = 0.0; // Assume evaluated at delta=0 for Jacobian
    //         double step_scale = 0.0;

    //         Eigen::Vector3d rot_axis = d.cross(n_proj).normalized();

    //         // ∂v/∂s = cos(0)*u1 + sin(0)*u = n_proj
    //         Eigen::Vector3d dn_dscale = n_proj;

    //         // ∂v/∂δγ = s*(-sin(0)*u1 + cos(0)*u) = norm_n * rot_axis
    //         Eigen::Vector3d dn_dtheta = norm_n * rot_axis;

    //         J.block<3, 1>(3 + 3 * i, 2 + 2 * i) = dn_dtheta;
    //         J.block<3, 1>(3 + 3 * i, 2 + 2 * i + 1) = dn_dscale;
    //     }
    // }

    return true;
}

bool LineParaRiemanLineParameterization::ComputeJacobian(const double *x,
                                                         double *jacobian) const
{
    Eigen::Map<const Eigen::Vector3d> d(x);
    Eigen::Matrix<double, 3, 2> d_basis = UtiliLine::TangentBasis(d);

    // std::cout << "ComputeJacobian d_basis:" << d_basis << std::endl;

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J(jacobian, GlobalSize(), LocalSize());
    J.setZero();

    // Direction Jacobian
    J.block<3, 2>(0, 0) = d_basis;

    // Normals Jacobian
    for (int i = 0; i < num_lines_; ++i)
    {
        Eigen::Matrix<double, 3, 2> n_basis = UtiliLine::TangentBasis(d);
        J.block<3, 2>(3 + 3 * i, 2 + 2 * i) = n_basis;
    }

    return true;

    /// ----------------------------------------------->
    // const int global_size = GlobalSize(); // 3 + 3 * num_lines_
    // const int local_size = LocalSize();   // 2 + 2 * num_lines_

    // Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J(jacobian, global_size, local_size);
    // J.setZero();

    // Eigen::Vector3d dir(x[0], x[1], x[2]);
    // dir.normalize();
    // Eigen::Matrix<double, 3, 2> basis = UtiliLine::TangentBasis(dir);

    // // Fill first 3 rows (for direction part)
    // J.block<3, 2>(0, 0) = basis;

    // // Fill 3x2 blocks for each line normal
    // for (int i = 0; i < num_lines_; ++i)
    // {
    //     int g_offset = 3 + 3 * i;
    //     int l_offset = 2 + 2 * i;
    //     J.block<3, 2>(g_offset, l_offset).setIdentity();
    // }

    // return true;
    /// ----------------------------------------------->
    //   const int global_size = GlobalSize(); // 3 * num_lines_+3
    //   const int local_size = LocalSize();   // 2 + 2 * num_lines_

    //   Eigen::Map<const Eigen::Vector3d> direction(x); // First 3 values are the direction
    //   Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J(jacobian, global_size, local_size);

    //   J.setZero();

    //   J.topRows(local_size).setIdentity();

    //   return true;
}

bool LineRiemanParameterization::ComputeJacobian(const double *x,
                                                 double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> J(jacobian);
    J.setZero();

    // Identity mapping from local 4D tangent vector to ambient 6D (approximate)
    for (int i = 0; i < 4; ++i)
        J(i, i) = 1.0;

    return true;
}

bool LineOrthParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    const double eps = 1e-8;
    double x_plus_delta[4];

    Eigen::Matrix<double, 4, 4> J;
    for (int i = 0; i < 4; ++i)
    {
        double delta[4] = {0.0, 0.0, 0.0, 0.0};
        delta[i] = eps;

        Plus(x, delta, x_plus_delta);

        for (int j = 0; j < 4; ++j)
        {
            J(j, i) = (x_plus_delta[j] - x[j]) / eps;
        }
    }

    Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> j_out(jacobian);
    j_out = J;
    return true;
}

bool ParaLineOrthParameterization::Plus(const double *x,
                                        const double *delta,
                                        double *x_plus_delta) const
{
    // theta --> U,  phi --> W
    double theta = *(x);   // theta varphi alpha
    double phi = *(x + 1); // beta
    // Vector3d theta = orth.head(3);
    // double phi = orth[3];
    double w1 = cos(phi);
    double w2 = sin(phi);

    // update
    double _delta_theta = *(delta);
    double _delta_phi = *(delta + 1);

    Eigen::Matrix2d W;
    W << w1, -w2, w2, w1;
    Eigen::Matrix2d delta_W;
    delta_W << cos(_delta_phi), -sin(_delta_phi), sin(_delta_phi),
        cos(_delta_phi);
    W = W * delta_W;

    // U' -- > theta'. W' --> phi'
    double *theta_pluse(x_plus_delta);

    double *phi_plus(x_plus_delta + 1);

    *theta_pluse = theta + _delta_theta;

    *phi_plus = asin(W(1, 0));

    return true;
}
bool ParaLineOrthParameterization::ComputeJacobian(const double *x,
                                                   double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> j(jacobian);
    j.setIdentity();

    return true;
}

// global
bool LineRiemanParameterizationGlobal::Plus(const double *x,
                                            const double *delta,
                                            double *x_plus_delta) const
{
    // Eigen::Map<const Eigen::Matrix<double, 6, 1>> plk(x);
    Eigen::Map<const Eigen::Vector3d> vanishing_direction(x);
    std::cout << "1. direction_: " << vanishing_direction.x() << "," << vanishing_direction.y() << "," << vanishing_direction.z() << std::endl;
    Eigen::Matrix<double, 3, 2> TangentBasisVectors = UtiliLine::TangentBasis(vanishing_direction);
    //---> update on manifold
    Eigen::Map<const Eigen::Vector2d> _delta_theta(delta);
    double _direct_step0 = *(delta);
    double _direct_step1 = *(delta + 1);

    Eigen::Vector3d _manifold_direc_move = _direct_step0 * TangentBasisVectors.col(0) + _direct_step1 * TangentBasisVectors.col(1);
    // from TS^2 to S2
    Eigen::Vector3d _direct = vanishing_direction * cos(_manifold_direc_move.norm()) + _manifold_direc_move / (_manifold_direc_move.norm()) * sin(_manifold_direc_move.norm());
    Eigen::Map<Eigen::Vector3d> theta_pluse(x_plus_delta);
    theta_pluse[0] = _direct.x();
    theta_pluse[1] = _direct.y();
    theta_pluse[2] = _direct.z();

    std::cout << "2. updated direction_: " << _direct.x() << "," << _direct.y() << "," << _direct.z() << std::endl;
    return true;
}

bool LineRiemanParameterizationGlobal::ComputeJacobian(const double *x,
                                                       double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 3, 2, Eigen::RowMajor>> j(jacobian);
    // j.setIdentity();
    j.topRows<2>().setIdentity();
    j.bottomRows<1>().setZero();
    return true;
}

LineRiemanParameterizationUnique::LineRiemanParameterizationUnique(const double *_vanishing_direction)
    // const double *_tangent_vectors,
    // const double * tengen[vanishing_direction_id])
    : vd_(_vanishing_direction) {};

bool LineRiemanParameterizationUnique::ComputeJacobian(const double *x,
                                                       double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 3, 2, Eigen::RowMajor>> j(jacobian);
    // j.setIdentity();
    j.topRows<2>().setIdentity();
    j.bottomRows<1>().setZero();
    return true;
}

bool LineRiemanParameterizationUnique::Plus(const double *x,
                                            const double *delta,
                                            double *x_plus_delta) const
{
    // Eigen::Map<const Eigen::Matrix<double, 6, 1>> plk(x);
    Eigen::Map<const Eigen::Vector3d> normal(x);

    Eigen::Map<const Eigen::Vector3d> direction_(vd_);
    // std::cout << "direction_ from outside:" << direction_ << std::endl;

    // Eigen::Matrix<double, 3, 2> TangentBasisVectors = UtiliLine::TangentBasis(direction_);
    double d = normal.norm();

    //---> update on manifold
    Eigen::Map<const Eigen::Vector2d> _delta_theta(delta);
    double _normal_step = *delta;
    // Eigen::Vector3d _manifold_direc_move = _direct_step0 * TangentBasisVectors.col(0) + _direct_step1 * TangentBasisVectors.col(1);
    // // from TS^2 to S2
    // Eigen::Vector3d _direct = plk.tail(3) * cos(_manifold_direc_move.norm()) + _manifold_direc_move / (_manifold_direc_move.norm()) * sin(_manifold_direc_move.norm());

    // align normal (TODO: use rotation)
    Eigen::Vector3d _normal = (normal - direction_ * (direction_.transpose() * normal)).normalized();

    std::cout << "********" << std::endl
              << normal.x() << ", " << normal.y() << "," << normal.z() << ". dot_product:" << direction_.transpose() * normal << std::endl;
    Eigen::Vector3d TangentVector = UtiliLine::TangentVector(direction_, _normal); // for normal

    Eigen::Vector3d _manifold_normal_move = _normal_step * TangentVector;
    // from TS to S
    // _normal = _normal * cos(_manifold_normal_move.norm()) + _manifold_normal_move / (_manifold_normal_move.norm()) * sin(_manifold_normal_move.norm());
    _normal = (_normal + _manifold_normal_move).normalized();
    // SO(2) update
    d += *(delta + 1);

    _normal = d * _normal;

    std::cout << "updated:" << _normal.x() << "," << _normal.y() << std::endl;

    Eigen::Map<Eigen::Matrix<double, 3, 1>>
        theta_pluse(x_plus_delta);
    theta_pluse[0] = _normal.x();
    theta_pluse[1] = _normal.y();
    theta_pluse[2] = _normal.z();
    return true;
}

bool LineZhaoLiangParameterization::Plus(const double *x,
                                         const double *delta,
                                         double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> plk(x);

    Eigen::Matrix<double, 3, 2> TangentBasisVectors = UtiliLine::TangentBasis(plk.tail(3));
    // Eigen::Vector3d TangentVector = UtiliLine::TangentVector(plk.tail(3), plk.head(3)); // for normal
    // Eigen::Vector4d LineOrth = UtiliLine::plk_to_orth(plk);
    // double phi = LineOrth[3];
    // double w1 = cos(phi);
    // double w2 = sin(phi);
    double d = plk.head(3).norm();

    //---> update on manifold
    Eigen::Map<const Eigen::Vector3d> _delta_theta(delta);
    double _direct_step0 = *(delta);
    double _direct_step1 = *(delta + 1);
    double _normal_step = *(delta + 2);
    Eigen::Vector3d _manifold_direc_move = _direct_step0 * TangentBasisVectors.col(0) + _direct_step1 * TangentBasisVectors.col(1);
    // from TS^2 to S2
    Eigen::Vector3d _direct = plk.tail(3) * cos(_manifold_direc_move.norm()) + _manifold_direc_move / (_manifold_direc_move.norm()) * sin(_manifold_direc_move.norm());

    // align normal (TODO: use rotation)
    Eigen::Vector3d _normal = (plk.head(3).normalized() - _direct * (_direct.transpose() * plk.head(3).normalized())).normalized();
    Eigen::Vector3d TangentVector = UtiliLine::TangentVector(_direct, _normal); // for normal

    Eigen::Vector3d _manifold_normal_move = _normal_step * TangentVector;
    // from TS to S
    _normal = _normal * cos(_manifold_normal_move.norm()) + _manifold_normal_move / (_manifold_normal_move.norm()) * sin(_manifold_normal_move.norm());
    // SO(2) update
    d += *(delta + 3);
    // Eigen::Matrix2d W;
    // W << w1, -w2, w2, w1;
    // Eigen::Matrix2d delta_W;
    // delta_W << cos(_delta_phi), -sin(_delta_phi), sin(_delta_phi),
    //     cos(_delta_phi);
    // W = W * delta_W;

    // //---> reback
    // phi = asin(W(1, 0));
    // w1 = cos(phi);
    // _normal = w1 * _normal;
    _normal = d * _normal;
    Eigen::Map<Eigen::Matrix<double, 6, 1>>
        theta_pluse(x_plus_delta);
    theta_pluse[3] = _direct.x();
    theta_pluse[4] = _direct.y();
    theta_pluse[5] = _direct.z();
    theta_pluse[0] = _normal.x();
    theta_pluse[1] = _normal.y();
    theta_pluse[2] = _normal.z();
    return true;
}

bool LineZhaoLiangParameterization::ComputeJacobian(const double *x,
                                                    double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> j(jacobian);
    // j.setIdentity();
    j.topRows<4>().setIdentity();
    j.bottomRows<2>().setZero();
    return true;
}
