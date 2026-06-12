/***
 * @Author: yanyan-li yanyan.li.camp@gmail.com
 * @Date: 2025-07-12 22:42:03
 * @LastEditTime: 2025-07-12 22:42:03
 * @LastEditors: yanyan-li yanyan.li.camp@gmail.com
 * @Description:
 * @FilePath: /RiemanLine/src/optimizer/CovisPointParaRiemanLine copy.hpp
 */

#ifndef __VENOM_SRC_OPTIMIZER_COVISEXTE_EXTEN_REIMANNIAN_FINAL_GRAPH_BUNDLEADJUSTMENT__
#define __VENOM_SRC_OPTIMIZER_COVISEXTE_EXTEN_REIMANNIAN_FINAL_GRAPH_BUNDLEADJUSTMENT__
#include "src/manager_env/EnvTrajectory.hpp"
#include "src/optimizer/factor/LineProjectionFactor.hpp"
#include "src/optimizer/factor/ParaRiemanlineProjectionFactor.hpp"
#include "src/optimizer/factor/RiemanLineProjectionFactor.hpp"

#include "glog/logging.h"
#include <ceres/ceres.h>
#include <ceres/dynamic_numeric_diff_cost_function.h>
#include <cstdlib> // for rand() and RAND_MAX
#include <iostream>
#include <unordered_set>

#include "src/manager_env/EnvManager.hpp"
#include "src/optimizer/parametrization/line_parameterization.hpp"
#include "src/optimizer/parametrization/pose_local_parameterization.hpp"
#include "src/utils/UtilTransformer.hpp"

namespace simulator
{
    namespace optimizer
    {
        class CovisPointParaRiemanLine
        {
        public:
            static void optimizer(
                simulator::Trajectory *ptr_robot_trajectory,
                simulator::OptimizationManager &opti_para,
                std::map<int /*mappoint_id*/, Eigen::Vector3d> &optimized_mappoints,
                std::map<int /*mapline_id*/, Eigen::Matrix<double, 3, 2>> &optimized_maplines,
                std::map<int, Eigen::Matrix4d> &optimized_poses)
            {
                // ==== set parameters ====
                constexpr size_t first_kf_id_in_map = 0;
                constexpr int opmization_times = 1;
                constexpr int max_iters = 5;
                constexpr int min_obser_num = 3;
                const float huber_thres = 1.5;
                const float pixel_sigma = 1.5;

                // ==== intrinsic ====
                double fx = ptr_robot_trajectory->cam_intri.fx;
                double fy = ptr_robot_trajectory->cam_intri.fy;
                double cx = ptr_robot_trajectory->cam_intri.cx;
                double cy = ptr_robot_trajectory->cam_intri.cy;

                // ==== construct ceres optim parameters ====
                ceres::Problem problem;
                ceres::LossFunction *loss_function = nullptr;
                ceres::ParameterBlockOrdering *ordering = new ceres::ParameterBlockOrdering();
                ceres::LocalParameterization *pose_parameterization = new PoseLocalParameterization();
                loss_function = new ceres::CauchyLoss(3.0 / fx);

                // ====== 1. add mappoints ======
                std::map<int, double[3]> Para_Point_Feature;
                for (auto mp = opti_para.mappoints.begin(); mp != opti_para.mappoints.end(); mp++)
                {
                    int mp_id = mp->first;
                    if (opti_para.mappoints[mp_id] == Vec3::Zero())
                        continue;
                    if (opti_para.mappoints[mp_id].norm() < 0.0001)
                        continue;
                    // TODO: it means the mp_id is less than mappoint.size
                    Para_Point_Feature[mp_id][0] = mp->second(0);
                    Para_Point_Feature[mp_id][1] = mp->second(1);
                    Para_Point_Feature[mp_id][2] = mp->second(2);
                    problem.AddParameterBlock(Para_Point_Feature[mp_id], 3);
                }

                // ====== 2. add camera pose ======
                double Para_Pose[opti_para.kfs.size()][7];
                for (auto kf = opti_para.kfs.begin(); kf != opti_para.kfs.end(); kf++)
                {
                    int kf_id = kf->first;
                    Eigen::Matrix4d Twc = kf->second;
                    Eigen::Matrix3d Rot = Twc.block(0, 0, 3, 3);
                    Eigen::Vector3d Tran = Twc.block(0, 3, 3, 1);
                    Eigen::Quaterniond qua(Rot);
                    Para_Pose[kf_id][0] = Tran(0);
                    Para_Pose[kf_id][1] = Tran(1);
                    Para_Pose[kf_id][2] = Tran(2);
                    Para_Pose[kf_id][3] = qua.x();
                    Para_Pose[kf_id][4] = qua.y();
                    Para_Pose[kf_id][5] = qua.z();
                    Para_Pose[kf_id][6] = qua.w();

                    problem.AddParameterBlock(Para_Pose[kf_id], 7, pose_parameterization);
                    if (kf_id == 0)
                        problem.SetParameterBlockConstant(Para_Pose[kf_id]);
                }

                // ====== 3. prepare single and parallel mapparaline ======
                std::map<int /*parali_id*/, std::vector<int /*mapline_id*/>> paralineid_mlids;
                std::unordered_set<int> single_line;
                // reset id of each vanishing point
                int vanishing_id = 0;
                for (auto &parali_id : opti_para.asso_paralineid_mlids)
                {
                    //---> build ParaLine
                    Eigen::Vector3d direction = Eigen::Vector3d::Zero();
                    std::vector<int> para_inlier;
                    std::vector<int> individual_line;
                    // ===== 3.1 vectors: parallel_lines and individual_lines =====
                    UtiliLine::parallel_candidate(parali_id.second, opti_para.maplines, para_inlier, individual_line);
                    // sets of paraline and ids
                    if (para_inlier.size() > 2)
                    {
                        paralineid_mlids[vanishing_id] = para_inlier;
                        vanishing_id += 1;
                    }

                    // std::cout << paralineid_mlids.size() << " sets of paralines." << std::endl;
                    // for (int i = 0; i < paralineid_mlids.size(); i++)
                    //     std::cout << "the " << i << " th set has " << paralineid_mlids[i].size() << " lines" << std::endl;

                    // std::cout << " the set has " << single_line.size() << " single lines" << std::endl;
                }

                // todo: I want to save these opti_para.maplines not in the ml_ids into single_line
                std::unordered_set<int> grouped_ml_ids;
                for (const auto &parali_entry : paralineid_mlids)
                {
                    for (int ml_id : parali_entry.second)
                        grouped_ml_ids.insert(ml_id);
                }
                for (size_t ml_id = 0; ml_id < opti_para.maplines.size(); ++ml_id)
                {
                    // Skip if ml_id is already grouped
                    if (grouped_ml_ids.count(ml_id))
                        continue;

                    const auto &line = opti_para.maplines[ml_id];
                    if (line == Eigen::Matrix<double, 3, 2>::Zero())
                        continue;

                    // single_line.push_back(static_cast<int>(ml_id));
                    single_line.insert(ml_id);
                }

                // based on the

                // ====== 4. add ParaRiemanian lines ======
                std::map<int, std::vector<double>> para_line_params;
                std::map<int, Eigen::Matrix<double, 6, 1>> Para_Line_Feature;
                std::map<int, int> asso_mlid_ceresblock; // ml_id -> local block offset
                int para_set_id = 0;

                for (const auto &parali_entry : paralineid_mlids)
                {
                    const auto &ml_ids = parali_entry.second;
                    int set_size = ml_ids.size();
                    if (set_size == 0)
                        continue;

                    std::vector<double> param(3 + 3 * set_size, 0.0); // [dir(3), normal_1(3), ..., normal_n(3)]
                    Eigen::Vector3d direction_sum = Eigen::Vector3d::Zero();
                    int valid_count = 0;

                    // Compute shared direction
                    for (const int ml_id : ml_ids)
                    {
                        const auto &line = opti_para.maplines[ml_id];
                        if (line == Eigen::Matrix<double, 3, 2>::Zero())
                            continue;

                        Eigen::Vector3d start = line.col(0);
                        Eigen::Vector3d end = line.col(1);
                        Eigen::Vector3d dir = (end - start).normalized();

                        if (valid_count > 0 && direction_sum.dot(dir) < 0)
                        {
                            dir = -dir;
                            assert(false); // this is impossible, since we have changed the direction
                        }

                        direction_sum += dir;
                        ++valid_count;
                    }

                    if (valid_count == 0)
                        continue;

                    Eigen::Vector3d direction = direction_sum.normalized();
                    param[0] = direction[0];
                    param[1] = direction[1];
                    param[2] = direction[2];

                    // Compute plane normals
                    int offset = 0;
                    for (const int ml_id : ml_ids)
                    {
                        const auto &line = opti_para.maplines[ml_id];
                        Eigen::Vector3d start = line.col(0);
                        Eigen::Vector3d end = line.col(1);
                        Eigen::Vector3d dir = end - start;
                        Eigen::Vector3d normal = start.cross(end);

                        double d = normal.norm() / dir.norm();
                        normal.normalize();

                        // update the 3D line based on the normal
                        Eigen::Vector3d orthogonal_component = normal - direction * (direction.dot(normal));
                        orthogonal_component = orthogonal_component.normalized();

                        // std::cout << "orthogonal_component:" << orthogonal_component << std::endl;
                        Eigen::Vector3d final_normal = d * orthogonal_component;

                        param[3 * offset + 3] = final_normal[0];
                        param[3 * offset + 4] = final_normal[1];
                        param[3 * offset + 5] = final_normal[2];

                        // std::cout << final_normal << "," << orthogonal_component << std::endl;

                        asso_mlid_ceresblock[ml_id] = offset;
                        ++offset;
                    }

                    // Register to Ceres
                    ceres::LocalParameterization *local_param = new ParaRiemanLineParameterization(set_size);
                    problem.AddParameterBlock(param.data(), 3 + 3 * set_size, local_param);
                    para_line_params[para_set_id++] = std::move(param);
                }

                for (const auto &ml : opti_para.maplines)
                {
                    int ml_id = ml.first;

                    if (!single_line.count(ml_id))
                        continue;

                    const Mat32 &line_endpoints = ml.second;

                    // Skip uninitialized lines
                    if (line_endpoints == Mat32::Zero())
                        continue;

                    // Check for invalid or too short line
                    double line_length = (line_endpoints.col(0) - line_endpoints.col(1)).norm();
                    if (line_length < 0.01)
                    {
                        std::cerr << "[Warning] Line " << ml_id << " too short.\n";
                        continue;
                    }

                    // Compute PLK representation
                    const Eigen::Vector3d &start = line_endpoints.col(0);
                    const Eigen::Vector3d &end = line_endpoints.col(1);
                    Eigen::Vector3d direction = (end - start).normalized();
                    Eigen::Vector3d normal = start.cross(end).normalized();
                    double d = (start.cross(end)).norm() / line_length;
                    // consistency direction @Yanyan
                    // if (normal.dot(direction) < 0)
                    //     normal = -normal;
                    Eigen::Matrix<double, 6, 1> LinePLK;
                    LinePLK << d * normal, direction;
                    // from plk to orth
                    // Eigen::Vector4d LineOrth = UtiliLine::plk_to_orth(LinePLK);
#if __VERBOSE__OFF
                    // check plk_to_orth  and  orth_to_plk
                    // maplines: 26.3424, 7.98188, 3.30027, 0.957031, 0.289986,3.30027, 27.5251, 27.5251
                    Eigen::Matrix<double, 6, 1> opti_ml_endpoints = UtiliLine::orth_to_plk(LineOrth);
                    Vector3d v = opti_ml_endpoints.head(3);
                    Vector3d n = opti_ml_endpoints.tail(3);
                    std::cout << "maplines: " << LinePLK.head(3)
                              << ", " << LinePLK.tail(3)
                              << ", " << LinePLK.head(3).norm() / LinePLK.tail(3).norm()
                              << ", " << v
                              << ", " << n
                              << "," << v.norm() / n.norm()
                              << ", " << LinePLK.head(3).norm() / v.norm()
                              << ", " << LinePLK.tail(3).norm() / n.norm()
                              << std::endl;
#endif
                    Para_Line_Feature[ml_id][0] = LinePLK(0);
                    Para_Line_Feature[ml_id][1] = LinePLK(1);
                    Para_Line_Feature[ml_id][2] = LinePLK(2);
                    Para_Line_Feature[ml_id][3] = LinePLK(3);
                    Para_Line_Feature[ml_id][4] = LinePLK(4);
                    Para_Line_Feature[ml_id][5] = LinePLK(5);

                    ceres::LocalParameterization *local_parameterization_line = new LineRiemanParameterization();
                    problem.AddParameterBlock(Para_Line_Feature[ml_id].data(), 6, local_parameterization_line); // p,q
                }

                // ====== 4.1. construct ceres problem pose ======
                for (auto asso_mp_mea = opti_para.asso_mp_meas.begin(); asso_mp_mea != opti_para.asso_mp_meas.end(); asso_mp_mea++)
                {
                    int mp_id = asso_mp_mea->first;
                    if (opti_para.mappoints[mp_id] == Vec3::Zero())
                        continue;
                    if (asso_mp_mea->second.size() < min_obser_num) // more than two frames detect this point
                        continue;
                    for (int i = 0; i < asso_mp_mea->second.size(); i++)
                    {
                        int kf_id_i = asso_mp_mea->second[i].first;
                        Eigen::Vector2d mp_obser_i = asso_mp_mea->second[i].second;
                        Eigen::Vector2d mp_obser_i_unit((mp_obser_i(0) - cx) / fx, (mp_obser_i(1) - cy) / fy);
                        PointProjectionFactor *f = new PointProjectionFactor(mp_obser_i_unit);
                        problem.AddResidualBlock(f, loss_function, Para_Pose[kf_id_i], Para_Point_Feature[mp_id]);
                    }
                }

                // ====== 4.2. construct ceres problem pose ======
                int set_para_id = 0;
                for (const auto &asso_mpl_mea : paralineid_mlids)
                {
                    int para_line_id = asso_mpl_mea.first;
                    const std::vector<int> &ml_ids = asso_mpl_mea.second;

                    if (ml_ids.size() < 2)
                    {
                        std::cerr << "[Error] Parallel line group " << para_line_id << " has fewer than 2 lines.\n";
                        assert(false);
                    }

                    int set_size = static_cast<int>(ml_ids.size());

                    for (int i = 0; i < set_size; ++i)
                    {
                        int ml_id = ml_ids[i];
                        if (opti_para.maplines[ml_id] == Mat32::Zero())
                            continue;

                        const auto &observations = opti_para.asso_ml_meas[ml_id];

                        for (const auto &ml_meas : observations)
                        {
                            int frame_id = ml_meas.first;
                            Eigen::Vector4d obs = ml_meas.second;

                            // Normalize to image coordinates
                            Eigen::Vector4d obs_unit(
                                (obs(0) - cx) / fx,
                                (obs(1) - cy) / fy,
                                (obs(2) - cx) / fx,
                                (obs(3) - cy) / fy);

                            int block_id = asso_mlid_ceresblock.at(ml_id); // safe with .at

                            auto *cost_function = new RiemanianlinesetProjectionFactorX(obs_unit, block_id, set_size);

                            problem.AddResidualBlock(
                                cost_function, loss_function,
                                Para_Pose[frame_id],
                                para_line_params[set_para_id].data());
                        }
                    }

                    ++set_para_id;
                }

                for (auto asso_ml_mea = opti_para.asso_ml_meas.begin(); asso_ml_mea != opti_para.asso_ml_meas.end(); asso_ml_mea++)
                {
                    int ml_id = asso_ml_mea->first;
                    if (opti_para.maplines[ml_id] == Mat32::Zero())
                        assert(1 == 0);

                    if (asso_ml_mea->second.size() < min_obser_num) // more than two frames detect this point
                        continue;

                    if (!single_line.count(ml_id))
                        continue;

                    for (int i = 0; i < asso_ml_mea->second.size(); i++)
                    {
                        int kf_id_i = asso_ml_mea->second[i].first;
                        Eigen::Vector4d ml_obser_i = asso_ml_mea->second[i].second;

                        Eigen::Vector4d mp_obser_i_unit(
                            (ml_obser_i(0) - cx) / fx,
                            (ml_obser_i(1) - cy) / fy,
                            (ml_obser_i(2) - cx) / fx,
                            (ml_obser_i(3) - cy) / fy);

                        RiemanLineProjectionFactor *f = new RiemanLineProjectionFactor(mp_obser_i_unit);
                        // RiemanGlobalLocalLineProjectionFactor *f = new RiemanGlobalLocalLineProjectionFactor(mp_obser_i_unit);

                        problem.AddResidualBlock(f, loss_function, Para_Pose[kf_id_i], Para_Line_Feature[ml_id].data());
                    }
                }

                // solve ceres problem
                ceres::Solver::Options options;
                // options.linear_solver_type = ceres::DENSE_SCHUR;
                options.minimizer_progress_to_stdout = true;
                options.max_num_iterations = 100;
                options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
                // options.max_trust_region_radius = 1.0; // 限制最大步长
                // options.min_trust_region_radius = 1e-6;
                // options.initial_trust_region_radius = 0.1;
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);

#if __VERBOSE__
                std::cout << summary.FullReport() << std::endl;
#endif

                // update ceres results
                for (auto kf = opti_para.kfs.begin(); kf != opti_para.kfs.end(); kf++)
                {
                    int kf_id = kf->first;
                    Eigen::Vector3d Tran(Para_Pose[kf_id][0], Para_Pose[kf_id][1], Para_Pose[kf_id][2]);
                    Eigen::Quaterniond qua(Para_Pose[kf_id][6], Para_Pose[kf_id][3], Para_Pose[kf_id][4], Para_Pose[kf_id][5]);
                    Eigen::Matrix3d Rot = qua.toRotationMatrix();
                    Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
                    Twc.block(0, 0, 3, 3) = Rot;
                    Twc.block(0, 3, 3, 1) = Tran;
                    // std::cout<<"pose: "<< kf->second<<std::endl<<Twc<<std::endl;
                    optimized_poses[kf_id] = Twc;
                }
                // update mappoints
                for (auto mp = opti_para.mappoints.begin(); mp != opti_para.mappoints.end(); mp++)
                {
                    int mp_id = mp->first;
                    Vec3 optimized_mappoint;
                    optimized_mappoint << Para_Point_Feature[mp_id][0], //= mp->second(0);
                        Para_Point_Feature[mp_id][1],                   //= mp->second(1);
                        Para_Point_Feature[mp_id][2];                   //= mp->second(2);
                    optimized_mappoints[mp_id] = optimized_mappoint;
                    // std::cout<<"original: "<< mp->second(0)<<","<< mp->second(1)<< ","<< mp->second(2)<<std::endl;
                    // std::cout << "optimized: " << optimized_mappoint(0) << "," << optimized_mappoint(1) << "," << optimized_mappoint(2) << std::endl;
                }

                // update maplines
                // === Update optimized maplines from Para_Line_1 ===
                for (const auto &parali_id : paralineid_mlids)
                {
                    int vd_id = parali_id.first;
                    const std::vector<int> &line_ids = parali_id.second;
                    int size_set = static_cast<int>(line_ids.size());

                    if (para_line_params.count(vd_id) == 0)
                    {
                        std::cerr << "[Error] Cannot find optimized parameters for vd_id=" << vd_id << std::endl;
                        continue;
                    }

                    const std::vector<double> &param = para_line_params.at(vd_id); // use set index as key

                    int counter = 1; // 0-2 for direction, then 3n for normals

                    for (int i = 0; i < size_set; ++i)
                    {
                        int ml_id = line_ids[i];
                        if (opti_para.maplines[ml_id] == Mat32::Zero())
                            continue;

                        Vector6d riemannian_line;
                        riemannian_line.tail<3>() = Eigen::Vector3d(param[0], param[1], param[2]);
                        riemannian_line.head<3>() = Eigen::Vector3d(
                            param[3 * counter + 0],
                            param[3 * counter + 1],
                            param[3 * counter + 2]);
                        counter++;

                        Eigen::Matrix<double, 6, 1> EndPoints3d = Eigen::Matrix<double, 6, 1>::Zero();
                        int valid_observations = 0;

                        for (const auto &obs : opti_para.asso_ml_meas[ml_id])
                        {
                            int kf_id_i = obs.first;
                            const Eigen::Vector4d &ml_obser = obs.second;

                            Eigen::Vector4d mp_obser_i_unit(
                                (ml_obser(0) - cx) / fx,
                                (ml_obser(1) - cy) / fy,
                                (ml_obser(2) - cx) / fx,
                                (ml_obser(3) - cy) / fy);

                            const Eigen::Matrix4d &Twc = optimized_poses[kf_id_i];
                            Vector6d EndPoints3d_i = UtiliLine::plk_to_endpoints(riemannian_line, mp_obser_i_unit, Twc);

                            double line_length = (EndPoints3d_i.head<3>() - EndPoints3d_i.tail<3>()).norm();
                            if (line_length < 0.1)
                                continue;

                            Eigen::Vector3d dist_s = EndPoints3d_i.head<3>() - EndPoints3d.head<3>();
                            Eigen::Vector3d dist_e = EndPoints3d_i.tail<3>() - EndPoints3d.head<3>();

                            if (dist_s.norm() < dist_e.norm())
                            {
                                EndPoints3d += EndPoints3d_i;
                            }
                            else
                            {
                                EndPoints3d.head<3>() += EndPoints3d_i.tail<3>();
                                EndPoints3d.tail<3>() += EndPoints3d_i.head<3>();
                            }
                            valid_observations++;
                        }

                        if (valid_observations > 0)
                        {
                            EndPoints3d /= valid_observations;
                            optimized_maplines[ml_id].col(0) = EndPoints3d.head<3>();
                            optimized_maplines[ml_id].col(1) = EndPoints3d.tail<3>();
                        }
                        else
                        {
                            std::cerr << "[Warning] No valid observations for ml_id = " << ml_id << std::endl;
                        }
                    }

                    // update maplines
                    for (auto asso_ml_mea = opti_para.asso_ml_meas.begin(); asso_ml_mea != opti_para.asso_ml_meas.end(); asso_ml_mea++)
                    {
                        //

                        int ml_id = asso_ml_mea->first;

                        if (!single_line.count(ml_id))
                            continue;
                        if (opti_para.maplines[ml_id] == Mat32::Zero())
                            assert(1 == 0);
                        if (asso_ml_mea->second.size() < min_obser_num) // more than two frames detect this point
                            continue;

                        // std::cout << "optimized_maplines_:" << std::endl;

                        Eigen::Matrix<double, 6, 1> optimized_mapline;
                        optimized_mapline << Para_Line_Feature[ml_id][0],
                            Para_Line_Feature[ml_id][1],
                            Para_Line_Feature[ml_id][2],
                            Para_Line_Feature[ml_id][3],
                            Para_Line_Feature[ml_id][4],
                            Para_Line_Feature[ml_id][5];

                        // Eigen::Vector3d endpoint_s = opti_para.maplines[ml_id].col(0);
                        // Eigen::Vector3d endpoint_e = opti_para.maplines[ml_id].col(1);

                        // Eigen::Matrix<double, 6, 1>
                        //     opti_ml_plk = UtiliLine::orth_to_plk(optimized_mapline_orth);

                        Vector6d endpoints = UtiliLine::plk_to_endpoints(optimized_mapline, opti_para.maplines[ml_id]);

                        optimized_maplines[ml_id].block(0, 0, 3, 1) = endpoints.head(3);
                        optimized_maplines[ml_id].block(0, 1, 3, 1) = endpoints.tail(3);

                        // std::cout << "ml_id:" << ml_id << "optimized:" << optimized_maplines[ml_id] << std::endl;
                    }
                }

                // std::cout << "optimized_maplines size: " << optimized_maplines.size() << std::endl;
                return;
            }
        };
    } // namespace optimizer
} // namespace simulator
#endif // __VENOM_SRC_OPTIMIZER_COVISEXTE_EXTEN_REIMANNIAN_FINAL_GRAPH_BUNDLEADJUSTMENT__